import torch.nn as nn


class SimpleCIFAR10(nn.Module):

    def __init__(self, activation='leaky_relu'):
        super().__init__()

        if activation == 'leaky_relu':
            act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            act,
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            act,
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x