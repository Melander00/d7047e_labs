from copy import deepcopy

import torch
from tqdm import tqdm


# Train and validation
def train_epoch(model, criterion, optimizer, train_loader, epoch_number, grad_scaler):
    model.train()

    device = next(model.parameters()).device

    total_loss=0
    correct=0
    guesses=0

    for batch_nr, (inputs, labels) in enumerate(tqdm(train_loader, leave=False)):
        optimizer.zero_grad()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type):
            prediction = model(inputs)
            loss = criterion(prediction, labels)
    
        total_loss += loss.item()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        guess = torch.argmax(prediction, dim=1)
        correct += (guess == labels).sum().item()
        guesses += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / guesses
    return avg_loss, accuracy


def validate_epoch(model, criterion, val_loader, epoch_number):
    model.eval()

    device = next(model.parameters()).device

    total_loss=0
    correct=0
    guesses=0

    with torch.no_grad():
        for batch_nr, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type):
                prediction = model(inputs)
                loss = criterion(prediction, labels)
                
            total_loss += loss.item()

            guess = torch.argmax(prediction, dim=1)
            correct += (guess == labels).sum().item()
            guesses += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / guesses
    return avg_loss, accuracy


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, show_epoch_output = True):

    # Move model to GPU if applicable
    device = next(model.parameters()).device

    grad_scaler = torch.GradScaler(device=device.type)

    best_val_loss = float("inf")
    best_model = deepcopy(model).cpu()

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, epoch, grad_scaler)
        val_loss, val_accuracy = validate_epoch(model, criterion, val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if show_epoch_output:
            print(f"\rEpoch [{epoch+1}/{num_epochs}]                                            \n \tLoss: Train={round(train_loss * 1000)/1000}; Val={round(val_loss * 1000) / 1000}\n \tAcc: Train={round(train_accuracy * 10000) / 100}%; Val={round(val_accuracy * 10000) / 100}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model).cpu()
    
    losses = (train_losses, val_losses)
    accuracies = (train_accuracies, val_accuracies)

    return best_model, losses, accuracies, model.cpu(), best_val_loss


# Testing
def test_model(model, criterion, test_loader):
    model.eval()
    
    device = next(model.parameters()).device

    total_loss=0
    
    correct=0
    guesses=0

    c_predictions = []
    c_labels = []

    with torch.no_grad():
        for batch_nr, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type):
                prediction = model(inputs)
                loss = criterion(prediction, labels)

            total_loss += loss.item()

            guess = torch.argmax(prediction, dim=1)
            correct += (guess == labels).sum().item()
            guesses += labels.size(0)

            c_predictions.append(guess)
            c_labels.append(labels)

    accuracy = correct / guesses if guesses > 0 else 0
    avg_loss = total_loss / len(test_loader)

    confusion_matrix = torch.zeros(1)

    # Flatten all mini-batches into single tensors
    all_predictions = torch.cat(c_predictions).to(device)
    all_labels = torch.cat(c_labels).to(device)

    # Find the number of classes
    num_classes = int(max(all_predictions.max(), all_labels.max()).item() + 1)

    # Build confusion matrix directly on GPU
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)
    confusion_matrix.index_put_(
        (all_labels, all_predictions),
        torch.ones_like(all_labels, dtype=confusion_matrix.dtype),
        accumulate=True
    )

    return accuracy, avg_loss, confusion_matrix