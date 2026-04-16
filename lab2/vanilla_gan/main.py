import json
import os
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from loaders import prepare_mnist_loaders
from models import Discriminator, Generator
from training import train_gan


def main():
    model_name = "vanilla_gan"
    version = "2"

    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100

    Z_dim = 1000
    h_dim = 128
    X_dim = 28 * 28
    
    loaders = prepare_mnist_loaders(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(Z_dim, h_dim, X_dim).to(device)
    D = Discriminator(X_dim, h_dim).to(device)

    criterion = nn.BCELoss()

    G_solver = optim.Adam(G.parameters(), lr=learning_rate)
    D_solver = optim.Adam(D.parameters(), lr=learning_rate)

    print("")
    print("="*6, f"Training {model_name}:{version}", "="*6)

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
        show_epoch_output=False,
        model_name=model_name,
        version=version,
    )

    end = time()

    metadata = {
        # ID
        "model_name": model_name,
        "version": version,
        
        # Hyperparameters
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "z_dim": Z_dim,
        "h_dim": h_dim,
        "X_dim": X_dim,

        # Training results
        "training_time": end - start,
        "best_g_loss": best_g_loss,
        "g_losses": G_losses,
        "d_losses": D_losses,
    }

    save_dir = f"./output/{model_name}/{version}"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    D = best_pair["D"]
    G = best_pair["G"]
    
    torch.save({
        "D": D.state_dict(),
        "G": G.state_dict()
    }, os.path.join(save_dir, "best_pair.pt"))



if __name__ == "__main__":
    main()