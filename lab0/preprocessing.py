import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def prepare_datasets(
    splits = [0.8,0.2]
):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.15),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    full_dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True)

    generator = torch.Generator().manual_seed(1)

    subsets = torch.utils.data.random_split(full_dataset, splits, generator=generator)

    train_dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transform_test)

    cifar_train = Subset(
        train_dataset,
        subsets[0].indices
    )

    cifar_val = Subset(
        val_dataset,
        subsets[1].indices
    )

    cifar_test = torchvision.datasets.CIFAR10(root=".", train=False, download=True, transform=transform_test)

    return cifar_train, cifar_val, cifar_test




def prepare_loaders(
    batch_size = 100,
    splits = [0.8,0.2]
):
    print("Preparing Data Loaders.")

    cifar_train, cifar_val, cifar_test = prepare_datasets(splits=splits)

    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_alexnet_loaders(
    batch_size = 100,
    splits = [0.8,0.2]
):
    print("Preparing Data Loaders.")

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.15),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    full_dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True)

    generator = torch.Generator().manual_seed(1)

    subsets = torch.utils.data.random_split(full_dataset, splits, generator=generator)

    train_dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transform_test)

    cifar_train = Subset(
        train_dataset,
        subsets[0].indices
    )

    cifar_val = Subset(
        val_dataset,
        subsets[1].indices
    )

    cifar_test = torchvision.datasets.CIFAR10(root=".", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
