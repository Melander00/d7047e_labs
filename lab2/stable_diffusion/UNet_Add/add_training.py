import torch
import numpy as np
from torch.optim import Adam
import tqdm
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from stable_diffusion.UNet_Add.add_unet import UNet_res
from stable_diffusion.dataloader import get_dataloader
from stable_diffusion.unet_modules import (
    Euler_Maruyama_sampler,
    get_diffusion_coeff_fn,
    get_marginal_prob_std_fn,
    loss_fn,
)


def train(n_epochs=75, batch_size=1024, lr=1e-3):

    # Load the MNIST dataset to a data loader
    data_loader = get_dataloader(batch_size=batch_size)[0]

    # Initialize the Adam optimizer with the specified learning rate.
    optimizer = Adam(score_model.parameters(), lr=lr)
    # Learning rate scheduler to adjust the learning rate during training.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))

    # Training loop over epochs.
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        # Iterate over mini-batches in the training data loader.
        for x, y in data_loader:
            x = x.to(device)
            # Compute the loss for the current mini-batch.
            loss = loss_fn(score_model, x, get_marginal_prob_std_fn())
            # Zero the gradients, backpropagate, and update the model parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate the total loss and the number of processed items.
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        
        # Adjust the learning rate using the scheduler.
        scheduler.step()
        lr_current = scheduler.get_last_lr()[0]
        
        # Print the average loss and learning rate for the current epoch.
        print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        
        # Save the model checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'ckpt_res.pth')
    
    
def visualize(sample_batch_size=64, num_steps=500):
    # Load the pre-trained model checkpoint
    ckpt = torch.load('ckpt_res.pth', map_location=device)
    score_model.load_state_dict(ckpt)
    
    # Choose the Euler-Maruyama sampler
    sampler = Euler_Maruyama_sampler

    # Generate samples using the specified sampler
    samples = sampler(score_model,
                    get_marginal_prob_std_fn(),
                    get_diffusion_coeff_fn(),
                    sample_batch_size,
                    num_steps=num_steps,
                    device=device,
                    y=None)

    # Clip samples to be in the range [0, 1]
    samples = samples.clamp(0.0, 1.0)

    # Visualize the generated samples
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    # Plot the sample grid
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()
    
if __name__ == "__main__":
    device = 'cuda'

    score_model = torch.nn.DataParallel(UNet_res(marginal_prob_std=get_marginal_prob_std_fn()))
    score_model = score_model.to(device)

    train()
    visualize()