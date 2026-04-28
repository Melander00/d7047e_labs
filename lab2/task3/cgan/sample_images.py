import torch
import torchvision
import matplotlib.pyplot as plt


def sample_images(G, Z_dim, device, save_path):
    """Generate one sample per digit class (0-9) and save as a labelled grid."""
    G.eval()

    z = torch.randn(10, Z_dim).to(device)
    labels = torch.arange(0, 10).to(device)

    with torch.no_grad():
        fake = G(z, labels).view(-1, 1, 28, 28).cpu()

    # Plain grid
    torchvision.utils.save_image(fake, save_path, nrow=10, normalize=True)

    # Annotated version with digit labels above each image
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, ax in enumerate(axes):
        img = fake[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_title(str(i), fontsize=12)
        ax.axis("off")
    plt.suptitle("cGAN samples — one per class", fontsize=13, y=1.05)
    plt.tight_layout()

    annotated_path = save_path.replace(".png", "_annotated.png")
    plt.savefig(annotated_path, bbox_inches="tight")
    plt.close()