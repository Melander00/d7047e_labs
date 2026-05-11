import torch
import torch.nn as nn
from torch import Tensor



import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Define layers here
        # Example:
        # self.layer = nn.Linear(in_features, out_features)
        self.encode=nn.Sequential(
            
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            

            
            )
        self.pool=nn.AdaptiveAvgPool2d((1,1))
   
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """

        # Example:
        # x = self.layer(x)
        x=self.encode(x)
        x=self.pool(x)
        x=torch.flatten(x,1)

        return x


