import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from skimage.io import imread
import cv2
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
data_directory=r"C:\Users\abhin\Desktop\Projects\CNN_PYTORCH\dataset"
dataset=ImageFolder(data_directory,transform=transforms.Compose([transforms.Resize((150,150)),transforms.ToTensor()]))
batch=128
train_size=int(len(dataset)*80/100)
test_size=int(len(dataset)*20/100)
train_data,test_data=random_split(dataset,[train_size,test_size])
train_dataloader=DataLoader(train_data,pin_memory=True,batch_size=batch,shuffle=True,num_workers=4)
test_dataloader=DataLoader(test_data,pin_memory=True,batch_size=batch*2,num_workers=4)
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
show_batch(train_dataloader)