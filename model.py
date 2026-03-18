import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from monai.data import  decollate_batch
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Lambdad,
    EnsureType,
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import wandb

class MLPAdapter(nn.Module):
    def __init__(self, dim, adapter_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        return x + residual

class SwinBlockAdapterWrapper(nn.Module):
    def __init__(self, block, dim,adapter_dim):
        super().__init__()
        self.block = block
        self.adapter = MLPAdapter(dim,adapter_dim)

    def forward(self, *args, **kwargs):
        x = self.block(*args, **kwargs)
        x = self.adapter(x)
        return x

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r, alpha):
        super().__init__()

        self.linear = linear_layer
        self.linear.weight.requires_grad = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        original = self.linear(x)
        lora = (x @ self.A.T @ self.B.T) * self.scaling
        return original + lora


class SwinBlockLoRAWrapper(nn.Module):
    def __init__(self, block, r, alpha):
        super().__init__()

        self.block = block

        # Replace qkv projection with LoRA
        self.block.attn.qkv = LoRALinear(
            self.block.attn.qkv,
            r=r,
            alpha=alpha
        )

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)

class Model(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.save_hyperparameters()
        set_determinism(seed=40)
        self.config=config
        self.mode=self.config.experiment.mode
        self.adapter_dim = config.experiment.get("adapter_dim", None)
        self.lora_rank=config.experiment.get("lora_rank",None)
        self.lora_alpha=config.experiment.get("lora_alpha",None)
        self.feature_size=self.config.model.feature_size
        self.in_channels=self.config.model.in_channels
        self.out_channels=self.config.model.out_channels
        self.use_checkpoint=self.config.model.use_checkpoint
        self.learning_rate=self.config.model.learning_rate
        self.weight_decay=self.config.model.weight_decay
        self.artifact_path=self.config.wandb.artifact_path
        self.model = SwinUNETR(
            spatial_dims=3,
            feature_size=self.feature_size,
            use_checkpoint=self.use_checkpoint,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )
        self.lossfunction=DiceLoss(to_onehot_y=True ,softmax=True,reduction='mean',include_background=False)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=self.out_channels)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=self.out_channels)])
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        if self.mode in ["fine tuning", "zero shot","adapter","lora"]:
            self._load_pretrained_from_wandb()
            if self.mode=="zero shot":
                self._freeze_model()
            if self.mode == "adapter":
                self._inject_adapters(adapter_dim=self.adapter_dim)
                self._freeze_backbone()
                self._print_trainable_params()
            if self.mode == "lora":
                self._inject_lora(r=self.lora_rank, alpha=self.lora_alpha)
                self._freeze_backbone()
                self._print_trainable_params()

    def configure_optimizers(self):
        if self.mode=="zero shot":
            return None
        if self.mode=="adapter":
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            return torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.mode == "lora":
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            return torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
    
    def forward(self,x):
        return self.model(x)
    
    def training_step(self,batch,batch_idx):
        inputs, labels =  batch["image"],batch["label"]
        outputs = self.forward(inputs)
        loss = self.lossfunction(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (128,128,128)
        sw_batch_size = 1
        with torch.no_grad():
            outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
            loss = self.lossfunction(outputs, labels)
            outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
            labels = [self.post_label(i) for i in decollate_batch(labels)]
            self.dice_metric(y_pred=outputs, y=labels)
            d = {"val_loss": loss.cpu().item(), "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d
  
    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"]
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate()[0].item()
        self.dice_metric.reset()
        mean_val_loss = val_loss / num_items
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        self.validation_step_outputs.clear()  # free memory
        self.log('Mean Validation Loss',mean_val_loss,prog_bar=True,logger=True)
        self.log('Mean Validation Dice',mean_val_dice,prog_bar=True,logger=True)
        return mean_val_loss,mean_val_dice
        
    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.input_img_size
        sw_batch_size = 1
        with torch.no_grad():
            outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
            outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
            labels = [self.post_label(i) for i in decollate_batch(labels)]
            dice=self.dice_metric(y_pred=outputs, y=labels)
            self.log("Test_Dice_Step",dice.item(),on_step=True,on_epoch=False,prog_bar=True,logger=True)
        return {"test_number": len(outputs)}

    def on_test_epoch_end(self):
        mean_test_dice = self.dice_metric.aggregate()[0].item()
        self.dice_metric.reset()
        self.log("Mean Test Dice", mean_test_dice, prog_bar=True)
        return mean_test_dice

    
    def _load_pretrained_from_wandb(self):
        api = wandb.Api()
        artifact = api.artifact(self.artifact_path, type="model")
        artifact_dir = artifact.download()
        ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
        ckpt_path = os.path.join(artifact_dir, ckpt_files[0])
        checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        print("✅ Pretrained weights loaded from WandB artifact.")

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad=False
        print("🔒 Model frozen (Zero Shot Mode)")

    def _inject_adapters(self, adapter_dim):
        swin = self.model.swinViT
        stages = [
        swin.layers1,
        swin.layers2,
        swin.layers3,
        swin.layers4]
        for stage in stages:                 # ModuleList
            for layer in stage:              # BasicLayer
                for i, block in enumerate(layer.blocks):   # SwinTransformerBlock
                    dim = block.mlp.linear1.in_features
                    layer.blocks[i] = SwinBlockAdapterWrapper(
                    block,
                    dim,
                    adapter_dim
                )
        print("✅ Adapters injected into Swin Transformer blocks")

    def _inject_lora(self, r, alpha):
        swin = self.model.swinViT
        stages = [
            swin.layers1,
            swin.layers2,
            swin.layers3,
            swin.layers4
        ]
        for stage in stages:
            for layer in stage:
                for i, block in enumerate(layer.blocks):
                    layer.blocks[i] = SwinBlockLoRAWrapper(
                    block,
                    r=r,
                    alpha=alpha
                    )
        print("✅ LoRA injected into Swin Transformer attention layers")

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if (
                "adapter" in name
                or "A" in name
                or "B" in name
                or "decoder" in name
                or "out" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("🔒 Backbone frozen, PEFT modules trainable")

    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable}")
        print(f"Total params: {total}")
        print(f"Trainable ratio: {trainable/total:.4f}")
