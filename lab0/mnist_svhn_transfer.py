"""
Task 0.2.2 — Transfer Learning from MNIST to SVHN

Stage 1: A CNN is trained from scratch on the MNIST handwritten digit dataset.
Stage 2: The convolutional feature extractor from Stage 1 is frozen and reused
         as a fixed backbone for classification on the Street View House Numbers
         (SVHN) dataset. Only the fully connected classifier head is retrained.

SVHN images are converted to grayscale to match the single-channel input
expected by the network trained on MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from mnist_model import MnistCNN
from model_trainer import test_model, train_model


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_mnist_loaders(batch_size: int = 128, splits: list = None):
    if splits is None:
        splits = [0.85, 0.15]

    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    generator = torch.Generator().manual_seed(42)
    subsets = torch.utils.data.random_split(full_train, splits, generator=generator)

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_eval)

    mnist_train = Subset(train_dataset, subsets[0].indices)
    mnist_val = Subset(val_dataset, subsets[1].indices)
    mnist_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_eval)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_svhn_loaders(batch_size: int = 128, splits: list = None):
    if splits is None:
        splits = [0.85, 0.15]

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train = torchvision.datasets.SVHN(root="./data", split="train", download=True)
    generator = torch.Generator().manual_seed(42)
    subsets = torch.utils.data.random_split(full_train, splits, generator=generator)

    train_dataset = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform_train)
    val_dataset = torchvision.datasets.SVHN(root="./data", split="train", download=True, transform=transform_eval)

    svhn_train = Subset(train_dataset, subsets[0].indices)
    svhn_val = Subset(val_dataset, subsets[1].indices)
    svhn_test = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=transform_eval)

    train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(svhn_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def run_training(model, loaders, criterion, optimizer, num_epochs):
    train_loader, val_loader, test_loader = loaders

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    best_model, losses, accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs
    )

    test_accuracy, confusion_matrix = test_model(best_model, test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    return best_model, losses, accuracies, test_accuracy


# ---------------------------------------------------------------------------
# Stage 1: Train on MNIST
# ---------------------------------------------------------------------------

def train_on_mnist(device, num_epochs: int = 20, batch_size: int = 128, lr: float = 1e-3):
    print("\n" + "=" * 60)
    print("Stage 1: Training CNN on MNIST")
    print("=" * 60)

    model = MnistCNN(num_classes=10).to(device)
    loaders = prepare_mnist_loaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model, losses, accuracies, test_accuracy = run_training(
        model, loaders, criterion, optimizer, num_epochs
    )

    writer = SummaryWriter(log_dir="runs/mnist_baseline")
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/Train", losses[0][epoch], epoch)
        writer.add_scalar("Accuracy/Train", accuracies[0][epoch], epoch)
        writer.add_scalar("Loss/Val", losses[1][epoch], epoch)
        writer.add_scalar("Accuracy/Val", accuracies[1][epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()

    return best_model, test_accuracy


# ---------------------------------------------------------------------------
# Stage 2: Transfer to SVHN (Feature Extraction)
# ---------------------------------------------------------------------------

def transfer_to_svhn(mnist_model, device, num_epochs: int = 20, batch_size: int = 128, lr: float = 1e-3):
    print("\n" + "=" * 60)
    print("Stage 2: Transfer Learning — MNIST → SVHN")
    print("Backbone frozen; only classifier head is trained.")
    print("=" * 60)

    model = MnistCNN(num_classes=10).to(device)
    model.load_state_dict(mnist_model.state_dict())

    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    loaders = prepare_svhn_loaders(batch_size=batch_size)

    best_model, losses, accuracies, test_accuracy = run_training(
        model, loaders, criterion, optimizer, num_epochs
    )

    writer = SummaryWriter(log_dir="runs/svhn_transfer")
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/Train", losses[0][epoch], epoch)
        writer.add_scalar("Accuracy/Train", accuracies[0][epoch], epoch)
        writer.add_scalar("Loss/Val", losses[1][epoch], epoch)
        writer.add_scalar("Accuracy/Val", accuracies[1][epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()

    return best_model, test_accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mnist_model, mnist_accuracy = train_on_mnist(
        device=device,
        num_epochs=20,
        batch_size=128,
        lr=1e-3,
    )

    svhn_model, svhn_accuracy = transfer_to_svhn(
        mnist_model=mnist_model,
        device=device,
        num_epochs=20,
        batch_size=128,
        lr=1e-3,
    )

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  MNIST  test accuracy : {mnist_accuracy * 100:.2f}%")
    print(f"  SVHN   test accuracy : {svhn_accuracy * 100:.2f}%")
    print("=" * 60)
