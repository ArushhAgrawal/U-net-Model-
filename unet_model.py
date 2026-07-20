import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import time
from PIL import Image
import tqdm
device= "mps" if torch.mps.is_available() else "cpu"
class DoubleConv(nn.Module):
    def __init__(self, in_features, out_features ):
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
    def __init__(self, in_channels=3,  out_channels=1, features=[64,128,256,512]):
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
    def __init__ (self, image_dir, mask_dir, resize_shape=(160,160), max_image=190):
        self.image_dir= image_dir
        self.mask_dir= mask_dir
        self.reshape= resize_shape
        all_images = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])#this sorts the list so computer doest do random input and messes the data
        self.images= all_images[:max_image]
    def __len__(self):
        return len(self.images)#returns the length so the epochs can be known when to end
    def __getitem__(self,index):
        image_name=self.images[index]
        image_path= os.path.join(self.image_dir, image_name)#creates a direct path to the image
        mask_name=image_name.replace(".jpg", ".png")#creating image mask file name
        mask_path= os.path.join(self.mask_dir, mask_name)#direct path to mask
        #loading image on computer memory
        image= Image.open(image_path).convert("RGB")
        mask= Image.open(mask_path).convert("L")
        #resizing image
        image= TF.resize(image,self.reshape)
        mask=TF.resize(mask,self.reshape, 
                       interpolation=TF.InterpolationMode.NEAREST)#using interpolation converts messy chunk into 0,1 cons- making it pixelated
        image_tensor= TF.to_tensor(image)
        mask_array= np.array(mask)
        mask_tensor= TF.to_tensor(mask)
        return image_tensor, mask_tensor
#loading data
full_data= ImageDataset(image_dir="data/train_image/",  mask_dir="data/train_mask", resize_shape=(160,160), max_image=190)
train_size= int(0.8*(len(full_data)))
val_size= len(full_data)-train_size
train_dataset, val_dataset= random_split(full_data, [train_size, val_size])
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=8,        
    shuffle=True)
val_loader= DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False)

#setting up device agnostic code
device= "mps" if torch.mps.is_available() else "cpu"
#defining model
model= UNET(in_channels=3, out_channels=1).to(device)

#training/eval
def train(loss_fn, epochs, model, optimizer, train_data, test_data):
    start= time.time()
    net_loss_train= []
    net_loss_test= []
    epoch_bar= tqdm.tqdm(range(epochs), desc= "Training U-Net")
    for epoch in epoch_bar:
        model.train()
        for batch, (x,y) in enumerate(train_data):
            x.to(device), y.to(device)
            y_train_logits= model(x.to(device))
            loss_train=loss_fn(y_train_logits,y.to(device))
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            net_loss_train.append(loss_train.item())
        model.eval()
        with torch.inference_mode():
            for batch, (x,y) in enumerate(test_data):
                y_logits_test= model(x.to(device))
                loss_test= loss_fn(y_logits_test, y.to(device)) 
                net_loss_test.append(loss_test.item())#why item is imp since if we dont do item it would be like this loss value, mps, gradfunc all this if we do item its just the loss value
    end=time.time()
    print("\ntime taken", end-start,"\nloss train", loss_train,"\nloss test", loss_test, "\nloss every batch train", net_loss_train, "\nloss every batch test",net_loss_test) 
optimizer= torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn= nn.BCEWithLogitsLoss()
model_train= train(loss_fn=loss_fn, epochs=3, model= model, optimizer=optimizer, train_data=train_loader, 
                   test_data= val_loader)
print(model_train)


