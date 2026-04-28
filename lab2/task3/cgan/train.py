import os

import torch
import torchvision
from tqdm import tqdm


def save_epoch_samples(G, Z_dim, device, save_dir, epoch):
    """Save a grid of 10 samples (one per class) for the current epoch."""
    G.eval()
    os.makedirs(save_dir, exist_ok=True)

    z = torch.randn(10, Z_dim).to(device)
    labels = torch.arange(0, 10).to(device)

    with torch.no_grad():
        fake = G(z, labels).view(-1, 1, 28, 28)

    torchvision.utils.save_image(
        fake,
        os.path.join(save_dir, f"{epoch}.png"),
        nrow=10,
        normalize=True,
    )
    G.train()


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
    sample_img_dir=None,    # e.g. "./output/cgan/50/img" — enables per-epoch saves
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

            # Label smoothing: 0.9 prevents D from becoming overconfident
            real_targets = torch.full((batch_size, 1), 0.9).to(device)
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

            g_loss = criterion(fake_logits, torch.ones(batch_size, 1).to(device))

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            pbar.set_postfix({"G": f"{g_loss.item():.4f}", "D": f"{d_loss.item():.4f}"})

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

        # Save one sample grid per epoch for TensorBoard visualisation
        if sample_img_dir is not None:
            save_epoch_samples(G, Z_dim, device, sample_img_dir, epoch)

    return best_pair, best_g_loss, G_losses, D_losses
