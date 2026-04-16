from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_mnist_loaders(
    batch_size = 64
):
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 image to 784
    ])

    train_dataset = datasets.MNIST(root='../MNIST', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return [train_loader]