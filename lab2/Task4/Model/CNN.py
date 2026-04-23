import torch




import torch
import torch.nn as nn




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(1,32,3,padding=1)
        self.c2=nn.Conv2d(32,16,3, padding=1)
        self.re=nn.ReLU()

        self.pool=nn.MaxPool2d(2)
        self.fc1=nn.Linear(16*7*7,10)






    def forward(self,x):
        x = self.pool(self.re(self.c1(x)))
        x = self.pool(self.re(self.c2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


        
