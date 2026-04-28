import torch
from torch.optim import Adam
import tqdm
from tqdm.auto import trange, tqdm
from torch.optim.lr_scheduler import LambdaLR
from time import time

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from stable_diffusion.UNet_Transformer.unet_transformer import UNet_Tranformer
from stable_diffusion.dataloader import get_dataloader
from stable_diffusion.unet_modules import (
    get_marginal_prob_std_fn,
    loss_fn_cond,
)
from stable_diffusion.UNet_Transformer.infer_unet_transformer import generate_grid


def train(n_epochs=101, batch_size=1024, lr=10e-4):
    best_dict = None #Used to store the state_dict of the best model checkpoint based on the lowest loss achieved during training.
    losses = [] #Used to store the average loss for each epoch during training, which can be useful for monitoring training progress and diagnosing issues.
    lr_list = [] #Used to store the learning rate for each epoch during training, which can be useful for monitoring how the learning rate changes over time, especially when using a learning rate scheduler.
    
    # Load the MNIST dataset to a data loader
    data_loader = get_dataloader(batch_size=batch_size)[0]

    # Define the optimizer and learning rate scheduler
    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

    start = time()

    best_loss = float('inf')
    # Use tqdm to display a progress bar over epochs
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0

        # Iterate over batches in the data loader
        for x, y in tqdm(data_loader):
            x = x.to(device)

            # Compute the loss using the conditional score-based model
            loss = loss_fn_cond(score_model, x, y, get_marginal_prob_std_fn())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        # Adjust learning rate using the scheduler
        scheduler.step()
        lr_current = scheduler.get_last_lr()[0]
        losses.append(avg_loss / num_items)
        lr_list.append(lr_current)

        # Print epoch information including average loss and current learning rate
        print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        if (avg_loss / num_items) < best_loss:
            best_loss = avg_loss / num_items
            best_dict = score_model.state_dict()
        
        # Save the model checkpoint after each epoch of training
        torch.save(score_model.state_dict(), 'ckpt_transformer.pth')
            
        if epoch % 10 == 0:
            out_dir = f"./output/unet_transformer/1/img"
            file_name = f"{epoch}.png"
            generate_grid(sample_batch_size=64, num_steps=250, digit=9, device=device, export_dir=out_dir, file_name=file_name)
        
    end = time()
    print(f"Training completed in {end - start:.2f} seconds.")
    
    metadata = {
        "model_name": "unet_transformer",
        "version": 1,
        "loss_function": "Denoising Conditional Score Matching Loss",
        "num_epochs": n_epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        #"z_dim": Z_dim,
        #"h_dim": h_dim,
        #"X_dim": X_dim,
        "training_time": end - start,
        "best_loss": best_loss,
        "losses": losses,
        "learning_rates": lr_list,
    }

    save_dir = f"./output/unet_transformer/1"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    # Save the best model checkpoint after training is complete
    torch.save(best_dict, 'ckpt_transformer.pth')
    
if __name__ == "__main__":
    device = 'cuda'

    score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=get_marginal_prob_std_fn()))
    score_model = score_model.to(device)

    train()