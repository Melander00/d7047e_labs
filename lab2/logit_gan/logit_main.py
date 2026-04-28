"""
Logit Loss GAN - Main Training Script (Task 2)

Runs 3 separate trainings: 5, 10, and 50 epochs.
Outputs saved to: ./output/logit_gan/{epochs}/

Key difference from vanilla GAN (main.py):
    - Uses Discriminator WITHOUT final sigmoid (from GAN_logit_loss.py)
    - Uses nn.BCEWithLogitsLoss() instead of nn.BCELoss()
"""

import json
import os
import sys
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

# Import shared training/inference/loaders from vanilla_gan folder
# This avoids code duplication and keeps vanilla_gan files untouched.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../vanilla_gan"))
from loaders import prepare_mnist_loaders
from training import train_gan

# Import the MODIFIED models from this folder (logit_gan)
from GAN_logit_loss import Discriminator, Generator


def run_logit_gan(num_epochs: int):
    """Train the logit GAN for a given number of epochs and save results."""

    model_name = "logit_gan"
    version = str(num_epochs)

    batch_size = 64
    learning_rate = 1e-3

    Z_dim = 1000
    h_dim = 128
    X_dim = 28 * 28

    loaders = prepare_mnist_loaders(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(Z_dim, h_dim, X_dim).to(device)
    D = Discriminator(X_dim, h_dim).to(device)

    # KEY CHANGE: BCEWithLogitsLoss instead of BCELoss
    # This is numerically stable and works with raw logits (no sigmoid in D)
    criterion = nn.BCEWithLogitsLoss()

    G_solver = optim.Adam(G.parameters(), lr=learning_rate)
    D_solver = optim.Adam(D.parameters(), lr=learning_rate)

    print(f"\n{'='*6} Training logit_gan for {num_epochs} epochs {'='*6}")
    print(f"Device: {device} | Loss: BCEWithLogitsLoss | Epochs: {num_epochs}")

    start = time()

    best_pair, best_g_loss, G_losses, D_losses = train_gan(
        generator=G,
        discriminator=D,
        generator_optim=G_solver,
        discriminator_optim=D_solver,
        criterion=criterion,
        num_epochs=num_epochs,
        loaders=loaders,
        Z_dim=Z_dim,
        show_epoch_output=True,
        model_name=model_name,
        version=version,
    )

    end = time()

    metadata = {
        "model_name": model_name,
        "version": version,
        "loss_function": "BCEWithLogitsLoss",
        "discriminator_sigmoid": False,
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

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    torch.save(
        {"D": best_pair["D"].state_dict(), "G": best_pair["G"].state_dict()},
        os.path.join(save_dir, "best_pair.pt"),
    )

    print(f"Done! Best G loss: {best_g_loss:.4f} | Time: {end - start:.1f}s")
    print(f"Outputs saved to: {save_dir}/")


def main():
    # Run for all three epoch counts as required by Task 2
    for epochs in [5, 10, 50, 100]:
        run_logit_gan(num_epochs=epochs)

    print("\n" + "="*40)
    print("All 3 runs complete!")
    print("Images saved at:  output/logit_gan/{5,10,50}/img/")
    print("Metadata saved at: output/logit_gan/{5,10,50}/metadata.json")
    print("="*40)


if __name__ == "__main__":
    main()
