import torch
from torchvision.utils import make_grid, save_image
import os
import numpy as np


import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from stable_diffusion.UNet_Transformer.unet_transformer import UNet_Tranformer
from stable_diffusion.dataloader import get_dataloader
from stable_diffusion.unet_modules import (
    Euler_Maruyama_sampler,
    get_diffusion_coeff_fn,
    get_marginal_prob_std_fn,
    loss_fn_cond,
)


def generate_grid(sample_batch_size=64, num_steps=250, digit=9, device='cuda', export_dir=None, file_name=None):
    ## Load the pre-trained checkpoint from disk.    
    score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=get_marginal_prob_std_fn()))
    score_model = score_model.to(device)
    ckpt = torch.load('ckpt_transformer.pth', map_location=device)
    score_model.load_state_dict(ckpt)

    # Choose the sampler type (Euler-Maruyama, pc_sampler, ode_sampler)
    sampler = Euler_Maruyama_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
    # score_model.eval()

    ## Generate samples using the specified sampler.
    samples = sampler(score_model,
            get_marginal_prob_std_fn(),
            get_diffusion_coeff_fn(),
            sample_batch_size,
            num_steps=num_steps,
            device=device,
            y=digit*torch.ones(sample_batch_size, dtype=torch.long))

    ## Sample visualization.
    samples = samples.clamp(0.0, 1.0)
    
    # Create a grid of samples for visualization
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

    if export_dir is not None:
        # Save the sample grid as an image file
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, file_name)
        save_image(sample_grid, export_path)
        print(f"Sample grid saved to {export_path}")
    else:
        # Plot the generated samples
        import matplotlib

        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        plt.show()
    
def start_inference():
    while True:
        digit = input("Enter a digit (0-9) to generate samples, or 'exit/quit/q' to quit: ")
        if digit.lower() in ['exit', 'quit', 'q']:
            break
        elif digit.isdigit() and 0 <= int(digit) <= 9:
            generate_grid(digit=int(digit))
        else:
            print("Invalid input. Please enter a digit between 0 and 9, or 'exit/quit/q' to quit.")
            
if __name__ == "__main__":
    try:
        start_inference()
    except KeyboardInterrupt:
        print("\nQuitting....")