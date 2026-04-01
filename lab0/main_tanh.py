"""
Task 0.1 — Tanh Activation Experiment

CNN trained on CIFAR-10 using Tanh activations and the Adam optimiser.
Results are logged to TensorBoard under runs/cifar10_tanh_adam.
"""

from multiprocessing import Process

import torch
import torch.nn as nn
import torch.optim as optim
from cifar_model import SimpleCIFAR10
from model_trainer import test_model, train_model
from preprocessing import prepare_loaders
from torch.utils.tensorboard import SummaryWriter


def sgd():
    learning_rate = 1e-2
    num_epochs = 100
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SimpleCIFAR10().to(device)
    loaders = prepare_loaders(batch_size=batch_size)
    train_loader, val_loader, test_loader = loaders

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_model, losses, accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs
    )

    test_accuracy, confusion_matrix = test_model(best_model, test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    writer = SummaryWriter(log_dir="runs/cifar10_sgd")
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/Train", losses[0][epoch], epoch)
        writer.add_scalar("Accuracy/Train", accuracies[0][epoch], epoch)
        writer.add_scalar("Loss/Val", losses[1][epoch], epoch)
        writer.add_scalar("Accuracy/Val", accuracies[1][epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()

def adam():
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SimpleCIFAR10().to(device)
    loaders = prepare_loaders(batch_size=batch_size)
    train_loader, val_loader, test_loader = loaders

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_model, losses, accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs
    )

    test_accuracy, confusion_matrix = test_model(best_model, test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    writer = SummaryWriter(log_dir="runs/cifar10_adam")
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/Train", losses[0][epoch], epoch)
        writer.add_scalar("Accuracy/Train", accuracies[0][epoch], epoch)
        writer.add_scalar("Loss/Val", losses[1][epoch], epoch)
        writer.add_scalar("Accuracy/Val", accuracies[1][epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()

def tanh():
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SimpleCIFAR10(activation="tanh").to(device)
    loaders = prepare_loaders(batch_size=batch_size)
    train_loader, val_loader, test_loader = loaders

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_model, losses, accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs
    )

    test_accuracy, confusion_matrix = test_model(best_model, test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix)

    writer = SummaryWriter(log_dir="runs/cifar10_tanh")
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/Train", losses[0][epoch], epoch)
        writer.add_scalar("Accuracy/Train", accuracies[0][epoch], epoch)
        writer.add_scalar("Loss/Val", losses[1][epoch], epoch)
        writer.add_scalar("Accuracy/Val", accuracies[1][epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()


if __name__ == "__main__":
    p1 = Process(target=sgd)
    p2 = Process(target=adam)
    p3 = Process(target=tanh)
    p1.start()
    p2.start()
    p3.start()