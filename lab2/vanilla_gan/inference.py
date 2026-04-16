import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from models import Generator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_generator_state(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/best_pair.pt"
    if not os.path.isfile(file_path):
        raise RuntimeError("Model and/or iteration does not exist on disk")

    return torch.load(file_path, weights_only=False)["G"]

def show_images(images, n=16):
    images = images[:n]
    images = images.view(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    model_name = "vanilla_gan"
    version = "1"

    Z_dim = 1000
    h_dim = 128
    X_dim = 28 * 28

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(Z_dim, h_dim, X_dim).to(device)

    # state = load_generator_state(model_name, version)
    # G.load_state_dict(state)
    G = load_generator_state(model_name, version).to(device)

    G.eval()

    num_samples = 16
    z = torch.randn(num_samples, Z_dim).to(device)

    with torch.no_grad():
        G_sample = G(z)

    G_sample = G_sample.cpu()
    show_images(G_sample)



def generate_images(
    G: nn.Module,
    out_dir: str,
    file_name: str,
    num_samples = 16,
    Z_dim = 1000,
):
    device = next(G.parameters()).device
    G.eval()

    z = torch.randn(num_samples, Z_dim).to(device)
    with torch.no_grad():
        samples = G(z).detach().cpu()
    
    images = samples[:num_samples]
    images = images.view(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, file_name), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()