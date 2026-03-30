import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from cifar_model import SimpleCIFAR10
from model_trainer import test_model, train_model
from preprocessing import prepare_alexnet_loaders, prepare_loaders
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def run_training(
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    criterion,
    optimizer,
    num_epochs=20,
):
    train_loader, val_loader, test_loader = loaders

    print(f"Starting training model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

    best_model, losses, accs = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=num_epochs
    )

    print("Training complete.")

    test_accuracy, confusion_matrix = test_model(best_model, test_loader=test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix)

    return best_model, losses, accs, test_accuracy, confusion_matrix


def main():

    learning_rate = 1e-4
    num_epochs = 30
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCIFAR10(activation="tanh").to(device)

    loaders = prepare_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model, losses, accuracies, test_accuracy, cm = run_training(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )

    writer = SummaryWriter(log_dir="runs/cifar10_tanh")

    for i in range(num_epochs):
        writer.add_scalar('Loss/Train', losses[0][i], i)
        writer.add_scalar('Accuracy/Train', accuracies[0][i], i)
        writer.add_scalar('Loss/Val', losses[1][i], i)
        writer.add_scalar('Accuracy/Val', accuracies[1][i], i)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()


def task2_ft():
    learning_rate = 1e-4
    num_epochs = 30
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, 10)

    model = model.to(device)

    loaders = prepare_alexnet_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model, losses, accuracies, test_accuracy, cm = run_training(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )

    writer = SummaryWriter(log_dir="runs/cifar10_alexnet_ft")

    for i in range(num_epochs):
        writer.add_scalar('Loss/Train', losses[0][i], i)
        writer.add_scalar('Accuracy/Train', accuracies[0][i], i)
        writer.add_scalar('Loss/Val', losses[1][i], i)
        writer.add_scalar('Accuracy/Val', accuracies[1][i], i)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()

def task2_fe():
    learning_rate = 1e-4
    num_epochs = 30
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 10)

    model = model.to(device)

    loaders = prepare_alexnet_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model, losses, accuracies, test_accuracy, cm = run_training(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )

    writer = SummaryWriter(log_dir="runs/cifar10_alexnet_fe")

    for i in range(num_epochs):
        writer.add_scalar('Loss/Train', losses[0][i], i)
        writer.add_scalar('Accuracy/Train', accuracies[0][i], i)
        writer.add_scalar('Loss/Val', losses[1][i], i)
        writer.add_scalar('Accuracy/Val', accuracies[1][i], i)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.close()

if __name__ == "__main__":
    task2_ft()
    task2_fe()