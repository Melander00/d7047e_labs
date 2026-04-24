import torch
from torch.optim import Adam
import tqdm
from tqdm.auto import trange, tqdm
from torch.optim.lr_scheduler import LambdaLR

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from stable_diffusion.UNet_Transformer.unet_transformer import UNet_Tranformer
from stable_diffusion.dataloader import get_dataloader
from stable_diffusion.unet_modules import (
    get_marginal_prob_std_fn,
    loss_fn_cond,
)


def train(n_epochs=100, batch_size=1024, lr=10e-4):

    # Load the MNIST dataset to a data loader
    data_loader = get_dataloader(batch_size=batch_size)[0]

    # Define the optimizer and learning rate scheduler
    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

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

        # Print epoch information including average loss and current learning rate
        print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        # Save the model checkpoint after each epoch of training
        torch.save(score_model.state_dict(), 'ckpt_transformer.pth')
    
if __name__ == "__main__":
    device = 'cuda'

    score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=get_marginal_prob_std_fn()))
    score_model = score_model.to(device)

    train()