"""
Conditional GAN (cGAN) - Main Training Script (Task 3)

Runs 3 separate trainings: 5, 10, and 50 epochs.
Outputs saved to: ./output/cgan/{epochs}/
"""

import json
import os
import sys
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

# Import shared loaders from vanilla_gan
from lab2.vanilla_gan.loaders import prepare_mnist_loaders

# Import your cGAN models + training loop
from lab2.task3.cgan.generator import Generator
from lab2.task3.cgan.discriminator import Discriminator
from lab2.task3.cgan.train import train_cgan

# Import plotting and sampling utilities
from lab2.task3.cgan.plot_losses import plot_losses
from lab2.task3.cgan.sample_images import sample_images


def run_cgan(num_epochs: int):
    """Train the cGAN for a given number of epochs and save results."""

    model_name = "cgan"
    version = str(num_epochs)

    batch_size = 64
    learning_rate = 1e-3

    Z_dim = 100
    h_dim = 128
    X_dim = 28 * 28

    # Unpack loaders and wrap into dictionary
    train_loader, test_loader = prepare_mnist_loaders(batch_size)

    loaders = {
        "train": train_loader,
        "test": test_loader
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(Z_dim, h_dim, X_dim).to(device)
    D = Discriminator(X_dim, h_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()

    G_solver = optim.Adam(G.parameters(), lr=learning_rate)
    D_solver = optim.Adam(D.parameters(), lr=learning_rate)

    print(f"\n===== Training cGAN for {num_epochs} epochs =====")
    print(f"Device: {device} | Loss: BCEWithLogitsLoss | Epochs: {num_epochs}")

    start = time()

    best_pair, best_g_loss, G_losses, D_losses = train_cgan(
        generator=G,
        discriminator=D,
        g_optimizer=G_solver,
        d_optimizer=D_solver,
        criterion=criterion,
        loaders=loaders,
        Z_dim=Z_dim,
        num_epochs=num_epochs,
        device=device,
        show_epoch_output=True,
    )

    end = time()

    metadata = {
        "model_name": model_name,
        "version": version,
        "loss_function": "BCEWithLogitsLoss",
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "z_dim": Z_dim,
        "h_dim": h_dim,
        "X_dim": X_dim,
        "training_time": end - start,
        "best_g_loss": best_g_loss,
        "g_losses": G_losses,
        "d_losses": D_losses,
    }

    save_dir = f"./output/{model_name}/{version}"
    os.makedirs(save_dir, exist_ok=True)

    # Save metadata
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Save best model weights
    torch.save(
        {"D": best_pair["D"], "G": best_pair["G"]},
        os.path.join(save_dir, "best_pair.pt"),
    )

    # Save loss plot
    plot_losses(
        G_losses,
        D_losses,
        save_path=os.path.join(save_dir, "losses.png")
    )

    # Save sample images
    sample_images(
        G,
        Z_dim,
        device,
        save_path=os.path.join(save_dir, "samples.png")
    )

    print(f"Done! Best G loss: {best_g_loss:.4f} | Time: {end - start:.1f}s")
    print(f"Outputs saved to: {save_dir}/")


def main():
    for epochs in [5, 10, 50]:
        run_cgan(num_epochs=epochs)

    print("\n========================================")
    print("All 3 cGAN runs complete!")
    print("Images saved at:  output/cgan/{5,10,50}/samples.png")
    print("Loss plots saved at: output/cgan/{5,10,50}/losses.png")
    print("Metadata saved at: output/cgan/{5,10,50}/metadata.json")
    print("========================================")


if __name__ == "__main__":
    main()