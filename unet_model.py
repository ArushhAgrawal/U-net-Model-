import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class DoubleConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.convstack= nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features), 
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features), 
            nn.ReLU()
        )
    def forward(self,x):
        return self.convstack(x)

class UNET(nn.Module):
    def __init__(self, in_channels,  out_channels, features=[64,128,256,512]):
        super().__init__()
        self.up= nn.ModuleList()
        self.down= nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size=2, stride=2)
        #down part
        for feature in features:
            self.down.append(DoubleConv(in_channels, feature))
            in_channels=feature
        
        #up part
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2,stride=2))#this will make the model move up
            self.up.append(DoubleConv(feature*2, feature))#this will do the conv part up 2 conv like this combined
        
        #bottom part
        self.bottom= DoubleConv(in_features=features[-1], out_features=features[-1]*2)#in feature then out feature 512*2
        self.final_conv= nn.Conv2d(features[0], out_channels,  kernel_size=1)
    def forward(self, x):
        skip=[]
        #downward pass
        for index in self.down:
            x=index(x)
            skip.append(x)#this is the part where it will skip the connection from the heightest to the lowest connection
            x=self.pool(x)#160x160-> 80x80
        x=self.bottom(x)
        skip= skip[::-1]#reversing the list 
        for index in range (0,len(self.up), 2):
            x=self.up[index](x)#up sampling
            skip_tensor= skip[index//2]#divides and then rounds to nearest whole number
            if skip_tensor.shape == x.shape:
                concat_skip= torch.cat((skip_tensor, x),dim=1)    
            else:
                l= x.shape[2]
                w=x.shape[3]
                concat_skip= torch.cat((skip_tensor[:,:, :l, :w], x),dim=1)
            x= self.up[index+1](concat_skip)
        return self.final_conv(x)

#test
# x=torch.randn((3,1,180,180))
# model= UNET(in_channels=1, out_channels=1)
# pred= model(x)
# print(pred.shape)
# print(x.shape)

#loading image dataset
class ImageDataset(Dataset):
    def __init__ (self, image_dir, mask_dir, resize_shape=(160,160), max_image=224):
        self.image_dir= image_dir
        self.mask_dir= mask_dir
        self.reshape= resize_shape

        all_images= sorted(os.listdir(image_dir))#this sorts the list so computer doest do random input and messes the data
        self.images= all_images[:max_image]
    def __len__(self):
        return len(self.image_dir)#returns the length so the epochs can be known when to end
    def __getitem__(self,index):
        image_name=self.images[index]
        image_path= os.path.join(self.image_dir, image_name)#creates a direct path to the image
        mask_name=image_name.replace(".jpg", ".png")#creating image mask file name
        mask_path= os.path.join(self.mask_dir, mask_name)#direct path to mask
        #loading image on computer memory
        image= Image.open(image_path).convert("RGB")
        mask= Image.open("mask_path").convert("RGB")
        #resizing image
        image= TF.resize(image,self.reshape)
        mask=TF.resize(image,self.reshape, 
                       interpolation=TF.InterpolationMode.NEAREST)#using interpolation converts messy chunk into 0,1 cons- making it pixelated
        image_tensor= TF.to_tensor(image_path)
        mask_array= np.array(mask)
        mask_tensor= torch.from_numpy(mask).float()
        mask_tensor= mask_tensor.unsqueeze(0)#converts from [h,w] to [1,h,w]
        return image_tensor, mask_tensor

Dataset= ImageDataset(image_dir="train_images",  mask_dir="train_masks", resize_shape=(160,160), max_image=224)
train_set, test_set= random_split(Dataset, [174,50])
train_loader= DataLoader(train_set, batch_size=4, shuffle=True)
test_loader= DataLoader(test_set, batch=4, shuffle=False)



