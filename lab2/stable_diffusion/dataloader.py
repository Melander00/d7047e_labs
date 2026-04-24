# Import the required libraries
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def get_dataloader(batch_size=64):
   # Load the MNIST dataset and create a data loader
    train_dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Download and load the test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader