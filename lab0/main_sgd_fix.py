"""
Task 0.1 — SGD Optimization Fix

This script addresses the "SGD not training" issue by using a more realistic 
learning rate (0.01) and adding momentum (0.9). These are standard practices 
to ensure SGD converges effectively on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cifar_model import SimpleCIFAR10
from model_trainer import test_model, train_model
from preprocessing import prepare_loaders


def main():
    # Improved Hyperparameters for SGD
    learning_rate = 1e-2  # Increased from 1e-4
    momentum = 0.9       # Added momentum to help convergence
    num_epochs = 30
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model with LeakyReLU (default)
    model = SimpleCIFAR10(activation="leaky_relu").to(device)
    
    # Prepare Data
    loaders = prepare_loaders(batch_size=batch_size)
    train_loader, val_loader, test_loader = loaders

    criterion = nn.CrossEntropyLoss()
    
    # SGD Optimizer with momentum
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print(f"Starting training with SGD (lr={learning_rate}, momentum={momentum})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_model, losses, accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs
    )

    # Final Evaluation
    test_accuracy, _ = test_model(best_model, test_loader=test_loader)
    print(f"\nFinal Test Accuracy (SGD): {test_accuracy * 100:.2f}%")

    # Logging to a new TensorBoard directory
    writer = SummaryWriter(log_dir="runs/cifar10_sgd_fixed")
    for epoch in range(num_epochs):
        writer.add_scalar("Loss/Train", losses[0][epoch], epoch)
        writer.add_scalar("Accuracy/Train", accuracies[0][epoch], epoch)
        writer.add_scalar("Loss/Val", losses[1][epoch], epoch)
        writer.add_scalar("Accuracy/Val", accuracies[1][epoch], epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()
    print("Results logged to runs/cifar10_sgd_fixed")


if __name__ == "__main__":
    main()
