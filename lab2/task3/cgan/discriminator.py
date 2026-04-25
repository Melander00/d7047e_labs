import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Conditional Discriminator for MNIST.
    Input: image x + label embedding
    Output: raw logit (no sigmoid)
    """
    def __init__(self, x_dim, h_dim, num_classes=10, label_emb_dim=16):
        super().__init__()

        # Embed class labels (0–9)
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # Input = flattened image + label embedding
        self.fc1 = nn.Linear(x_dim + label_emb_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

        self.apply(xavier_init)

    def forward(self, x, labels):
        """
        x:      (batch, 784) flattened MNIST image
        labels: (batch,) long tensor with class IDs
        """
        label_vec = self.label_emb(labels)          # (batch, label_emb_dim)
        d_in = torch.cat([x, label_vec], dim=1)     # concat conditioning
        h = F.relu(self.fc1(d_in))
        out = self.fc2(h)                           # raw logit
        return out