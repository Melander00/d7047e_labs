from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from inference import generate_images
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optim: optim.Optimizer,
    discriminator_optim: optim.Optimizer,
    criterion, # what type?
    loaders: list[DataLoader],
    model_name: str,
    version: str | int,
    num_epochs = 100,
    Z_dim = 1000,
    show_epoch_output = True,
):
    device = next(generator.parameters()).device

    grad_scaler = torch.GradScaler(device=device.type)

    best_g_loss = float("inf")
    best_pair = deepcopy({
        "G": generator.cpu(),
        "D": discriminator.cpu(), 
    })

    G_losses = []
    D_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training", leave=False, disable=show_epoch_output, unit="epoch"):
        G, D, G_loss_avg, D_loss_avg = train_gan_epoch(
            G=generator,
            D=discriminator,
            G_solver=generator_optim,
            D_solver=discriminator_optim,
            criterion=criterion,
            train_loader=loaders[0],
            grad_scaler=grad_scaler,
            Z_dim=Z_dim
        )

        G_losses.append(G_loss_avg)
        D_losses.append(D_loss_avg)

        if show_epoch_output:
            print(f"Epoch [{epoch+1}/{num_epochs}]\n \tLoss: G={G_loss_avg:.2f}; D={D_loss_avg:.2f}")


        if G_loss_avg < best_g_loss:
            best_g_loss = G_loss_avg
            best_pair = deepcopy({
                "G": G.cpu(),
                "D": D.cpu()
            })

        out_dir = f"./output/{model_name}/{version}/img"
        file_name = f"{epoch}.png"

        generate_images(
            G=G,
            out_dir=out_dir,
            file_name=file_name,
            Z_dim=Z_dim,
            num_samples=16,
        )

    return best_pair, best_g_loss, G_losses, D_losses


def train_gan_epoch(
    G: nn.Module,
    D: nn.Module,
    G_solver: optim.Optimizer,
    D_solver: optim.Optimizer,
    criterion, # what type?,
    train_loader: DataLoader,
    grad_scaler: torch.GradScaler,
    Z_dim: int = 1000
):
    G.train()
    D.train()

    device = next(G.parameters()).device

    D_loss_real_total = 0
    D_loss_fake_total = 0
    G_loss_total = 0

    for batch_nr, (inputs, labels) in enumerate(tqdm(train_loader, leave=False, desc="Epoch")):
        # Prepare real data
        X_real = inputs.float().to(device)

        # Sample noise and labels
        z = torch.randn(X_real.size(0), Z_dim).to(device)
        ones_label = torch.ones(X_real.size(0), 1).to(device)
        zeros_label = torch.zeros(X_real.size(0), 1).to(device)


        with torch.autocast(device_type=device.type):
            G_sample = G(z)
            D_real = D(X_real)
            D_fake = D(G_sample.detach())

            D_loss_real = criterion(D_real, ones_label)
            D_loss_fake = criterion(D_fake, zeros_label)

        D_loss = D_loss_real + D_loss_fake
        D_loss_real_total += D_loss_real.item()
        D_loss_fake_total += D_loss_fake.item()

        D_solver.zero_grad()
        grad_scaler.scale(D_loss).backward()
        grad_scaler.step(D_solver)
        grad_scaler.update()

        z = torch.randn(X_real.size(0), Z_dim).to(device)

        with torch.autocast(device_type=device.type):
            G_sample = G(z)
            D_fake = D(G_sample)
            G_loss = criterion(D_fake, ones_label)

        G_loss_total += G_loss.item()

        G_solver.zero_grad()
        grad_scaler.scale(G_loss).backward()
        grad_scaler.step(G_solver)
        grad_scaler.update()

    D_loss_real_avg = D_loss_real_total / len(train_loader)
    D_loss_fake_avg = D_loss_fake_total / len(train_loader)
    D_loss_avg = D_loss_real_avg + D_loss_fake_avg
    G_loss_avg = G_loss_total / len(train_loader)

    return G, D, G_loss_avg, D_loss_avg