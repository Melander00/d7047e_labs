import torch
from torch import nn


class simpleANN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # define layers here later
        print("simple ANN constructed")
        self.seq=nn.Sequential (
            nn.Linear(vocab_size,64),
            
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            
           
            nn.Linear(64, 2),
        )
        

    def forward(self, x):
        return self.seq(x)
        # define forward pass later
        


