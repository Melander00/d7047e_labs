import matplotlib.pyplot as plt
import torch
from torch import Tensor


def visualize_image(image: Tensor):
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()