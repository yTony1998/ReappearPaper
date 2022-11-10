from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, Dataset
from torchvision import datasets, transforms


train_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(256),
        transforms.ToTensor(),
    ]
)

class LoadDataset(Dataset):
    def __init__(self, file_list, transform =None):
        self.file_list = file_list
        self.transform = transform
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0
        
        return img_transformed, label
