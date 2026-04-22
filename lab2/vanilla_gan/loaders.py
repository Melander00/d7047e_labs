from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_mnist_loaders(batch_size=64):
    # Transform: convert to tensor + flatten 28x28 → 784
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Train dataset
    train_dataset = datasets.MNIST(
        root='../MNIST',
        train=True,
        transform=transform,
        download=True
    )

    # Test dataset
    test_dataset = datasets.MNIST(
        root='../MNIST',
        train=False,
        transform=transform,
        download=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    return train_loader, test_loader