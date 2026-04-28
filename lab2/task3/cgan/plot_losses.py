import matplotlib.pyplot as plt


def plot_losses(G_losses, D_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="D target (~0.5)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("cGAN Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
