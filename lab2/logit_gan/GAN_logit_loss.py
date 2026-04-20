import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Logit Loss GAN - Modified from vanilla_gan/models.py

Key Architectural Change:
    - Discriminator: REMOVED the final torch.sigmoid() activation.
      The Discriminator now outputs raw logits instead of probabilities [0,1].

    - Generator: UNCHANGED. The sigmoid output is still correct for generating
      pixel values in [0,1] for the MNIST images.

Why remove sigmoid?
    BCEWithLogitsLoss (logit loss) applies sigmoid INTERNALLY in a numerically
    stable way using the log-sum-exp trick. If we applied sigmoid before feeding
    into BCEWithLogitsLoss, we would be double-applying it, which would break
    training. We also avoid the gradient saturation problem that occurs when
    sigmoid squashes values to near 0 or 1 and kills the gradient signal.
"""


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """Unchanged from vanilla GAN — sigmoid output is correct for image pixels."""
    def __init__(self, z_dim, h_dim, x_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        out = torch.sigmoid(self.fc2(h))
        return out


class Discriminator(nn.Module):
    """
    MODIFIED: Final sigmoid removed — outputs raw logits.
    Must be used with nn.BCEWithLogitsLoss(), NOT nn.BCELoss().
    """
    def __init__(self, x_dim, h_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)       # <-- Raw logits, NO sigmoid here
        return out
