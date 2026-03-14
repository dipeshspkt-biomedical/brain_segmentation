import data
import model
import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class Train(object):
    def __init__(self,key,Save_Image_Directory,Save_Label_Directory,config):
        super().__init__()
        self.key=key
        self.img_dir=Save_Image_Directory
        self.label_dir=Save_Label_Directory
        self.config=config
        self.mode=self.config.experiment.mode.lower()
        self.log_dir="/kaggle/working/logs"
        self.project_name=self.config.wandb.project_name
        self.run_name=self.config.wandb.run_name
        os.makedirs(self.log_dir,exist_ok=True)
        wandb.login(key=self.key)
        self.wandb_logger=pl.loggers.WandbLogger(project=self.project_name, name=self.run_name,log_model='all')

    def train(self):
        DATA=data.MyData(self.img_dir,self.label_dir,self.config)
        MODEL=model.Model(self.config)
        checkpoint_callback = None
        if self.mode in ["scratch", "fine tuning","peft"]:
            checkpoint_callback = ModelCheckpoint(
                monitor='Mean Validation Dice',
                mode='max',
                save_top_k=1,
                dirpath='/kaggle/working/',
                filename='best-dice-model',
                save_last=True,
                save_weights_only=False
            )
        if self.mode == "zero shot":
            trainer = pl.Trainer(
                devices=[0],
                logger=wandb_logger,
            )
            print("Running Zero-Shot Evaluation...")
            trainer.test(model=MODEL, datamodule=DATA)
        else:
            trainer = pl.Trainer(
                devices=[0],
                max_epochs=self.config.train.max_epochs,
                logger=self.wandb_logger,
                log_every_n_steps=1,
                check_val_every_n_epoch=3,
                callbacks=[checkpoint_callback] if checkpoint_callback else None,
            )
            if self.mode=="fine tuning":
                print("Starting Full Fine Tuning")
            elif self.mode=="peft":
                print("Performing PEFT with adaptors")
            else:
                print('Training from Scratch')
            trainer.fit(model=MODEL, datamodule=DATA)

        
        
        
        
