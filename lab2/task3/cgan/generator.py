import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """
    Conditional Generator for MNIST.
    Kept shallow (2 layers, h_dim=64) so D has a fighting chance.
    Input: noise z + label embedding
    Output: flattened image (784) with values in [0, 1]
    """
    def __init__(self, z_dim, h_dim, x_dim, num_classes=10, label_emb_dim=16):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        self.fc1 = nn.Linear(z_dim + label_emb_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)

        self.apply(xavier_init)

    def forward(self, z, labels):
        """
        z:      (batch, z_dim)
        labels: (batch,) long tensor with class IDs
        """
        label_vec = self.label_emb(labels)
        x = torch.cat([z, label_vec], dim=1)
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out