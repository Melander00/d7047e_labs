import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from tqdm import tqdm


def train_cgan(
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    criterion,
    loaders,
    Z_dim,
    num_epochs,
    device,
    show_epoch_output=True,
):
    G = generator
    D = discriminator

    train_loader = loaders["train"]

    G_losses = []
    D_losses = []

    best_g_loss = float("inf")
    best_pair = {"G": None, "D": None}

    for epoch in range(1, num_epochs + 1):
        G.train()
        D.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.view(real_imgs.size(0), -1).to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)

            # -------------------------
            # Train Discriminator
            # -------------------------
            z = torch.randn(batch_size, Z_dim).to(device)
            fake_imgs = G(z, labels)

            real_targets = torch.ones(batch_size, 1).to(device)
            fake_targets = torch.zeros(batch_size, 1).to(device)

            real_logits = D(real_imgs, labels)
            fake_logits = D(fake_imgs.detach(), labels)

            d_loss_real = criterion(real_logits, real_targets)
            d_loss_fake = criterion(fake_logits, fake_targets)
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # -------------------------
            # Train Generator
            # -------------------------
            z = torch.randn(batch_size, Z_dim).to(device)
            fake_imgs = G(z, labels)
            fake_logits = D(fake_imgs, labels)

            g_loss = criterion(fake_logits, real_targets)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            # Update progress bar text
            pbar.set_postfix({"G": g_loss.item(), "D": d_loss.item()})

        # Average losses
        epoch_g_loss /= len(train_loader)
        epoch_d_loss /= len(train_loader)

        G_losses.append(epoch_g_loss)
        D_losses.append(epoch_d_loss)

        if show_epoch_output:
            print(f"Epoch {epoch}/{num_epochs} | G: {epoch_g_loss:.4f} | D: {epoch_d_loss:.4f}")

        if epoch_g_loss < best_g_loss:
            best_g_loss = epoch_g_loss
            best_pair["G"] = G.state_dict()
            best_pair["D"] = D.state_dict()

    return best_pair, best_g_loss, G_losses, D_losses