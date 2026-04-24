import torch
import numpy as np
from torch.optim import Adam
import tqdm
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from stable_diffusion.UNet_Concat.concat_unet import UNet
from stable_diffusion.dataloader import get_dataloader
from stable_diffusion.unet_modules import (
    Euler_Maruyama_sampler,
    get_diffusion_coeff_fn,
    get_marginal_prob_std_fn,
    loss_fn,
)


def train(n_epochs=50, batch_size=2048, lr=5e-4):

    # Load the MNIST dataset to a data loader
    data_loader = get_dataloader(batch_size=batch_size)[0]

    # Define the Adam optimizer for training the model
    optimizer = Adam(score_model.parameters(), lr=lr)

    # Progress bar for epochs
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        # Iterate through mini-batches in the data loader
        for x, y in tqdm(data_loader):
            x = x.to(device)
            # Calculate the loss and perform backpropagation
            loss = loss_fn(score_model, x, get_marginal_prob_std_fn())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss for the current epoch
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Save the model checkpoint after each epoch of training
        torch.save(score_model.state_dict(), 'ckpt.pth')
    
    
def visualize(sample_batch_size=64, num_steps=500):
    # Load the pre-trained model checkpoint
    ckpt = torch.load('ckpt.pth', map_location=device)
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

    score_model = torch.nn.DataParallel(UNet(marginal_prob_std=get_marginal_prob_std_fn()))
    score_model = score_model.to(device)

    train()
    visualize()