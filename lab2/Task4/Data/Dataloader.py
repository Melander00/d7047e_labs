
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np


#used for consitant split

seed = 42
generator = torch.Generator().manual_seed(seed)








def dataloader(batch_size:int=64):
    #downloads the Mnist dataset if not present and transforms it into tensors
    transform=transforms.ToTensor()

    Maindata = datasets.MNIST(
    root='Task4/Data',
    train=True,
    download=True,
    transform=transform
            )
    Testdata = datasets.MNIST(
    root='Task4/Data',
    train=False,
    download=True,
    transform=transform

    )
#splits the data into trian and val data.

    train_dataset, val_dataset = random_split(
    Maindata,
    [50000, 10000],
    generator=generator
    )
#---------------------------------Create dataloaders
    train_loader=DataLoader(batch_size=batch_size ,shuffle=True,dataset=train_dataset)
    val_loader=DataLoader(batch_size=batch_size ,shuffle=False,dataset=val_dataset)
    test_loader=DataLoader(batch_size=batch_size ,shuffle=False,dataset=Testdata)


    
     
    return train_loader, val_loader, test_loader
     



#TODO Load Mnist and return it as a dataloader