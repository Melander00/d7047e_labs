import json
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training.model_trainer import test_model, train_model


def run_training(
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    criterion,
    optimizer,
    model_name,
    num_epochs,
):
    train_loader, val_loader, test_loader = loaders

    device = next(model.parameters()).device

  
    print(f"Training {model_name} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters.")
    print(f"Running training on {device}")

    start_time = time.time()

    best_model, losses, accs, last_model, best_val_loss = train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=num_epochs
    )

    end_time = time.time()

    elapsed_time_seconds = end_time - start_time
    
    print(f"Finished training {model_name}")

    return best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds

def run_test(
    best_model: nn.Module,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    criterion,
    model_name,
):
    _,_,test_loader = loaders

    print(f"Testing the best version of {model_name}...")
    test_accuracy, test_loss, confusion_matrix = test_model(best_model, criterion, test_loader=test_loader)
    print(f"Test Accuracy for {model_name}: {test_accuracy:.4f}")

    return test_accuracy, test_loss, confusion_matrix







def run_model(
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    criterion,
    optimizer,
    model_name,
    num_epochs,
):
    device = next(model.parameters()).device


    best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds = run_training(
        model, loaders, criterion, optimizer, model_name, num_epochs
    )


    test_accuracy, test_loss, confusion_matrix = run_test(
        best_model.to(device), loaders, criterion, model_name
    )


    metadata = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "training_time": elapsed_time_seconds,
        "best_val_loss": best_val_loss,

        "train_loss": losses[0],
        "train_accuracy": accs[0],
        "val_loss": losses[1],
        "val_accuracy": accs[1],

        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": confusion_matrix,
    }

    return metadata, best_model, last_model





def save_model(
    output_dir,
    metadata,
    last_model,
    optimizer,
    best_model=None,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    if best_model:
        torch.save({
            "model_state": best_model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, os.path.join(output_dir, "best_model.pt"))

    torch.save({
        "model_state": last_model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, os.path.join(output_dir, f"last_model.pt"))

    






def develop_model(
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    criterion,
    optimizer,
    model_name,
    num_epochs,
    iteration_number = 0
):
    """
    Trains and test the model. Also saves the needed information to continue training later as well as some useful metadata.

    Returns the metadata. See GUIDELINES.md for schema.
    """
    
    print("")
    print("="*10, f"Developing {model_name}", "="*10)

    metadata, best_model, last_model = run_model(model, loaders, criterion, optimizer, model_name, num_epochs)

    print(f"Saving {model_name}:{iteration_number}")
    output_dir = f"./output/{model_name}/{iteration_number}"
    save_model(
        output_dir=output_dir,
        metadata=metadata,
        last_model=last_model,
        optimizer=optimizer,
        best_model=best_model
    )
    print("Development complete\n")
    return metadata





def continue_model_training(
    model: nn.Module,
    optimizer,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    model_name,
    iteration_number,
    criterion,
    num_epochs,
    new_learning_rate = None,
    load_optimizer = True,
):
    """
    Continues the training of a model. Remember to provide EXACT iteration_number and model_name.

    Returns the metadata. See GUIDELINES.md for schema.
    """

    device = next(model.parameters()).device

    output_dir = f"./output/{model_name}/{iteration_number}"

    states = torch.load(
        os.path.join(output_dir, "last_model.pt"), 
        map_location=next(model.parameters()).device
    )

    model.load_state_dict(states['model_state'])
    optimizer.load_state_dict(states['optimizer_state'])

    if new_learning_rate:
        for p in optimizer.param_groups:
            p["lr"] = new_learning_rate

    print("")
    print("="*10, f"Continuing {model_name}", "="*10)

    best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds = run_training(
        model.to(device), loaders, criterion, optimizer, model_name, num_epochs
    )

    with open(os.path.join(output_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    test_accuracy = metadata['test_accuracy']
    test_loss = metadata['test_loss']
    confusion_matrix = metadata['confusion_matrix']

    old_best_loss = metadata['best_val_loss']
    if best_val_loss < old_best_loss:
        print("New best model!")
        # Run tests since we have a new best model.
        test_accuracy, test_loss, confusion_matrix = run_test(
            best_model.to(device), loaders, criterion, model_name
        )
    else:
        # Don't overwrite the saved best model
        best_model = None

    updated_metadata = {
        "model_name": model_name,
        "num_epochs": metadata['num_epochs'] + num_epochs,
        "training_time": metadata['training_time'] + elapsed_time_seconds,
        "best_val_loss": min(best_val_loss, old_best_loss),

        "train_loss": list(metadata['train_loss']) + list(losses[0]),
        "train_accuracy": list(metadata['train_accuracy']) + list(accs[0]),
        "val_loss": list(metadata['val_loss']) + list(losses[1]),
        "val_accuracy": list(metadata['val_accuracy']) + list(accs[1]),

        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": confusion_matrix,
    }

    save_model(
        output_dir=output_dir,
        metadata=updated_metadata,
        last_model=last_model,
        optimizer=optimizer,
        best_model=best_model
    )

    print("Continuation complete.")
    return updated_metadata