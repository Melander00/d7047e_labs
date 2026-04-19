import torch
import torch.nn as nn
import torch.nn.functional as F

"""

Xavier init:

Ensures gradient doesnt explode or vanish by setting the weights from a distribution with zero-mean.
This prevents signal saturation by keeping activation variance consistent between layers.
Especially useful for sigmoid.

"""

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

"""

Generator & Discriminator:

Generator will learn to create images that look more and more like the dataset.
It outputs a generated image that the discriminator then will try and discern whether it was a real or fake image.
The generator uses random noise for these fake images.

Generator tries to fool Discriminator.
Discriminator tries to not be fooled.

When we have trained these models back and forth we will eventually have
- A generator that produces realistic images
- A discriminator that is uncertain, outputs ~0.5 (50% sure)

The training will be done once the discriminator converges at 0.5 loss.

"""

class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, num_classes=10):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, z_dim)
        self.fc1 = nn.Linear(z_dim + z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, z, labels):
        label_embed = self.embed(labels)
        x = torch.cat([z, label_embed], dim=1)
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out
    


class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim, num_classes=10):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(num_classes, x_dim)
        self.fc1 = nn.Linear(x_dim + x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x, labels):
        label_embed = self.embed(labels)
        x = torch.cat([x, label_embed], dim=1)
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out