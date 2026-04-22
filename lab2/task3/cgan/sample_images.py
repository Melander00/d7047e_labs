import torch
import torchvision

def sample_images(G, Z_dim, device, save_path):
    G.eval()
    z = torch.randn(10, Z_dim).to(device)
    labels = torch.arange(0, 10).to(device)

    with torch.no_grad():
        fake = G(z, labels).view(-1, 1, 28, 28)

    torchvision.utils.save_image(fake, save_path, nrow=10, normalize=True)