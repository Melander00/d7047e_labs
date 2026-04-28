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
    Stronger than G: 3 layers (256 -> 128 -> 1) to keep up
    with real images without collapsing.
    Input: image x + label embedding
    Output: raw logit (no sigmoid — pairs with BCEWithLogitsLoss)
    """
    def __init__(self, x_dim, h_dim, num_classes=10, label_emb_dim=16):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # Deeper D: 256 -> 128 -> 1
        # h_dim arg is ignored in favour of fixed 256/128 to ensure
        # D is always stronger than the 64-unit G
        self.fc1 = nn.Linear(x_dim + label_emb_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.apply(xavier_init)

    def forward(self, x, labels):
        """
        x:      (batch, 784) flattened MNIST image
        labels: (batch,) long tensor with class IDs
        """
        label_vec = self.label_emb(labels)
        d_in = torch.cat([x, label_vec], dim=1)
        h = F.relu(self.fc1(d_in))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)                   # raw logit
        return out