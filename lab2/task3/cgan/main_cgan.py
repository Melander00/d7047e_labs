"""
Conditional GAN (cGAN) — Main Training Script (Task 3)

Architecture:
  Generator:     z(100) + label_emb(16) -> 64 -> 784   (shallow/weak)
  Discriminator: x(784) + label_emb(16) -> 256 -> 128 -> 1  (deeper/strong)

Runs 3 separate trainings: 5, 10, and 50 epochs.
Outputs saved to: ./output/cgan/{epochs}/

To visualise with TensorBoard after training:
    python -m lab2.task3.cgan.tensorboard_writer
    tensorboard --logdir=./runs/cgan
"""

import json
import os
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from lab2.vanilla_gan.loaders import prepare_mnist_loaders

from lab2.task3.cgan.generator import Generator
from lab2.task3.cgan.discriminator import Discriminator
from lab2.task3.cgan.train import train_cgan
from lab2.task3.cgan.plot_losses import plot_losses
from lab2.task3.cgan.sample_images import sample_images


def run_cgan(num_epochs: int):
    """Train the cGAN for a given number of epochs and save results."""

    model_name = "cgan"
    version = str(num_epochs)

    batch_size = 64
    z_dim = 100
    x_dim = 28 * 28
    num_classes = 10
    label_emb_dim = 16

    # Asymmetric capacity: weak G vs strong D
    h_dim_g = 64    # G hidden dim  — intentionally small
    h_dim_d = 256   # D first layer — intentionally large (see discriminator.py)

    lr_g = 1e-3
    lr_d = 5e-4

    train_loader, test_loader = prepare_mnist_loaders(batch_size)
    loaders = {"train": train_loader, "test": test_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(
        z_dim=z_dim,
        h_dim=h_dim_g,
        x_dim=x_dim,
        num_classes=num_classes,
        label_emb_dim=label_emb_dim,
    ).to(device)

    D = Discriminator(
        x_dim=x_dim,
        h_dim=h_dim_d,
        num_classes=num_classes,
        label_emb_dim=label_emb_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    G_solver = optim.Adam(G.parameters(), lr=lr_g)
    D_solver = optim.Adam(D.parameters(), lr=lr_d)

    save_dir = f"./output/{model_name}/{version}"
    os.makedirs(save_dir, exist_ok=True)

    # Per-epoch sample images saved here for TensorBoard
    sample_img_dir = os.path.join(save_dir, "img")

    print(f"\n===== Training cGAN for {num_epochs} epochs =====")
    print(f"Device: {device} | lr_G={lr_g} | lr_D={lr_d} | Epochs: {num_epochs}")
    print(f"G: z({z_dim})+emb({label_emb_dim}) -> {h_dim_g} -> {x_dim}")
    print(f"D: x({x_dim})+emb({label_emb_dim}) -> 256 -> 128 -> 1")

    start = time()

    best_pair, best_g_loss, G_losses, D_losses = train_cgan(
        generator=G,
        discriminator=D,
        g_optimizer=G_solver,
        d_optimizer=D_solver,
        criterion=criterion,
        loaders=loaders,
        Z_dim=z_dim,
        num_epochs=num_epochs,
        device=device,
        show_epoch_output=True,
        sample_img_dir=sample_img_dir,
    )

    end = time()

    metadata = {
        "model_name": model_name,
        "version": version,
        "loss_function": "BCEWithLogitsLoss",
        "num_epochs": num_epochs,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "batch_size": batch_size,
        "z_dim": z_dim,
        "h_dim_g": h_dim_g,
        "h_dim_d": h_dim_d,
        "x_dim": x_dim,
        "training_time": end - start,
        "best_g_loss": best_g_loss,
        "g_losses": G_losses,
        "d_losses": D_losses,
    }

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    torch.save(
        {"G": best_pair["G"], "D": best_pair["D"]},
        os.path.join(save_dir, "best_pair.pt"),
    )

    plot_losses(G_losses, D_losses, save_path=os.path.join(save_dir, "losses.png"))
    sample_images(G, z_dim, device, save_path=os.path.join(save_dir, "samples.png"))

    print(f"Done! Best G loss: {best_g_loss:.4f} | Time: {end - start:.1f}s")
    print(f"Outputs saved to: {save_dir}/")


def main():
    for epochs in [5, 10, 50, 100]:
        run_cgan(num_epochs=epochs)

    print("\n========================================")
    print("All 3 cGAN runs complete!")
    print("Images saved at:     output/cgan/{5,10,50,100}/samples.png")
    print("Loss plots saved at: output/cgan/{5,10,50,100}/losses.png")
    print("Metadata saved at:   output/cgan/{5,10,50,100}/metadata.json")
    print("Per-epoch imgs at:   output/cgan/{5,10,50,100}/img/{epoch}.png")
    print("\nTo launch TensorBoard:")
    print("  python -m lab2.task3.cgan.tensorboard_writer")
    print("  tensorboard --logdir=./runs/cgan")
    print("========================================")


if __name__ == "__main__":
    main()