
import os
from torchvision import  transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from PIL import Image
import json,torch
class CustomDataset(Dataset):
    def __init__(self, split):
        with open('image_list.json', 'r') as f:
            self.split_list=json.load(f)[split]
        
        self.split = split
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30,interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
        self.totenor=transforms.ToTensor()
        self.preprocess=transforms.Compose([
            transforms.Resize((224,224)),
        ])
    def __getitem__(self, idx):
        data_path = self.split_list[idx]
        img = Image.open(data_path).convert('RGB')
        
        img = self.preprocess(img)
        if self.split == "train":
            img=self.transforms(img)
            
        # Convert mask and pos_embed to tensor
        img = self.img_transforms(img)
        
        image_name=os.path.basename(data_path)
        if image_name.startswith("re_"):
            class_label=1
        else:
            class_label=0
        return img, class_label


    def __len__(self):
        return len(self.split_list)