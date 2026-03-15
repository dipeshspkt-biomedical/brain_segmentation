import pytorch_lightning as pl
from monai.data import DataLoader, Dataset, decollate_batch,SmartCacheDataset,CacheDataset
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    CropForegroundd,
    Resized,
    ScaleIntensityd,
    CenterSpatialCropd,
    ResizeWithPadOrCropd,
    Orientation,
    DivisiblePadd,
    SpatialPadd,
    RandSpatialCropSamplesd,
    ToTensord,
    Lambdad,
    EnsureType,
)
from sklearn.model_selection import train_test_split
import file_handler
from file_handler import Files
import feature_extractor



class MyData(pl.LightningDataModule):
    def __init__(self,Save_Image_Directory,Save_Label_Directory,config):
        super().__init__()
        self.config=config
        self.Save_Image_Directory=Save_Image_Directory
        self.Save_Label_Directory=Save_Label_Directory
        self.data_source=self.config.data.data_source.lower()
        self.segmentation_mode=self.config.experiment.segmentation_mode.lower()
        self.mode=self.config.experiment.mode.lower()
        self.batch_size=self.config.data.batch_size
        self.train_size=self.config.data.train_size
        self.test_size=self.config.data.test_size
        self.random_state=self.config.data.random_state
        self.cache_rate=self.config.data.cache_rate
        self.num_workers=self.config.data.num_workers
        self.spatial_size=self.config.data.spatial_size
        self.train_transform=Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Lambdad(keys=['label'],func=lambda x: self.label_transformation(x,data_source=self.data_source,segm_mode=self.segmentation_mode)),
                Lambdad(keys=["image","label"],func=lambda x: self.change_orientation(x,data_source=self.data_source)),
                CropForegroundd(keys=['image','label'],source_key='label',select_fn=lambda x: x > 0),
                SpatialPadd(keys=['image','label'],spatial_size=self.spatial_size),
                ScaleIntensityd(keys=["image"]),  
                ToTensord(keys=["image","label"]),
                
            ]
        )
        self.val_transform=Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Lambdad(keys=['label'],func=lambda x: self.label_transformation(x,data_source=self.data_source,segm_mode=self.segmentation_mode)),
                Lambdad(keys=["image","label"],func=lambda x: self.change_orientation(x,data_source=self.data_source)),
                CropForegroundd(keys=['image','label'],source_key='label',select_fn=lambda x: x > 0),
                SpatialPadd(keys=['image','label'],spatial_size=self.spatial_size),
                ScaleIntensityd(keys=["image"]),  
                ToTensord(keys=["image","label"]),
                
            ]
        )
        self.test_transform=Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Lambdad(keys=['label'],func=lambda x: self.label_transformation(x,data_source=self.data_source,segm_mode=self.segmentation_mode)),
                Lambdad(keys=["image","label"],func=lambda x: self.change_orientation(x,data_source=self.data_source)),
                CropForegroundd(keys=['image','label'],source_key='label',select_fn=lambda x: x > 0),
                SpatialPadd(keys=['image','label'],spatial_size=self.spatial_size),
                ScaleIntensityd(keys=["image"]),  
                ToTensord(keys=["image","label"]),
                
            ]
        )

    def setup(self,stage=None):
        self.img_files=Files().extract_image_path(self.Save_Image_Directory)
        self.label_files=Files().extract_label_path(self.Save_Label_Directory)
        self.data_dict=[{'image':img,'label':label} for img,label in zip(self.img_files,self.label_files)]
        if self.mode in ["scratch","fine tuning","adapter","lora"]:
            self.train_files, self.val_files = train_test_split(self.data_dict,train_size=self.train_size,test_size=self.test_size,random_state=self.random_state)
            self.train_dataset=CacheDataset(data=self.train_files,transform=self.train_transform,cache_rate=self.cache_rate)
            self.val_dataset=CacheDataset(data=self.val_files,transform=self.val_transform,cache_rate=self.cache_rate)
            self.test_dataset=None
        elif self.mode=="zero shot":
            self.train_dataset=None
            self.val_dataset=None
            self.test_dataset=CacheDataset(data=self.data_dict,transform=self.test_transform,cache_rate=self.cache_rate)
            
  
    def train_dataloader(self):
        if self.mode=="zero shot":
            return None
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):
        if self.mode=="zero shot":
            return None
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

    @staticmethod
    def label_transformation(label,data_source,segm_mode):
        if data_source=="oasis":
            if segm_mode=="hippocampus":
                label[(label==17)|(label==53)]=1000
                label[label!=1000]=0
                label[label==1000]=1
            elif segm_mode=="whole":
                label=feature_extractor.Features().transform_label_values(label)   
        elif data_source=="tuth":
            label[(label==1)|(label==2)]=1000
            label[label!=1000]=0
            label[label==1000]=1  
        return label

    @staticmethod
    def change_orientation(x,data_source):
        if data_source in ["harp","tuth"]:
            orient=Orientation(axcodes="LIA")
            X=orient(x)
            return X
        return x
