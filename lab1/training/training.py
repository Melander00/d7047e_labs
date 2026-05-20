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

    

def _save_tensorboard_embeddings(output_dir: str, model: nn.Module, loader, device):
    target_layer = None
    if hasattr(model, "fc"):
        target_layer = model.fc
    else:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                target_layer = module

    if target_layer is None:
        print("No linear layer found for embedding extraction; skipping projector.")
        return

    captured_embeddings = []

    def _hook(module, input, output):
        captured_embeddings.append(input[0].detach().cpu())

    hook = target_layer.register_forward_hook(_hook)

    model.eval()
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            _ = model(inputs)
            labels_list.append(labels.detach().cpu())

    hook.remove()

    if len(captured_embeddings) == 0:
        print("No embeddings captured (empty loader?).")
        return

    embeddings = torch.cat(captured_embeddings, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    metadata_str = []
    raw_dataset = getattr(loader, "dataset", None)
    subset_indices = getattr(raw_dataset, "indices", None)
    base_dataset = getattr(raw_dataset, "dataset", None)
    if subset_indices is not None and hasattr(base_dataset, "get_raw_item"):
        for subset_idx in subset_indices:
            text, _ = base_dataset.get_raw_item(subset_idx)
            metadata_str.append(text.replace("\t", " ").replace("\n", " "))
    else:
        sentiment_map = {0: "negative", 1: "positive"}
        metadata_str = [sentiment_map.get(label.item(), str(label.item())) for label in labels_tensor]

    tb_logdir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_logdir)
    writer.add_embedding(mat=embeddings, metadata=metadata_str, tag="cls_embeddings", global_step=0)
    writer.flush()
    writer.close()

    with open(os.path.join(tb_logdir, "metadata.tsv"), "w", encoding="utf-8") as f:
        f.write("sentence\n")
        for sentence in metadata_str:
            f.write(f"{sentence}\n")

    torch.save({"embeddings": embeddings, "labels": labels_tensor}, os.path.join(tb_logdir, "embeddings.pt"))


def _save_word_embeddings(output_dir: str, embeddings_tensor: torch.Tensor, tokens, tag: str = "word_embeddings"):
    tb_logdir = os.path.join(output_dir, "tensorboard_words")
    os.makedirs(tb_logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_logdir)
    writer.add_embedding(mat=embeddings_tensor, metadata=list(tokens), tag=tag, global_step=0)
    writer.flush()
    writer.close()

    with open(os.path.join(tb_logdir, "metadata.tsv"), "w", encoding="utf-8") as f:
        f.write("word\n")
        for token in tokens:
            f.write(f"{token}\n")


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
    tensorboard_dir = f"./runs/{model_name}/{iteration_number}"
    save_model(
        output_dir=output_dir,
        metadata=metadata,
        last_model=last_model,
        optimizer=optimizer,
        best_model=best_model
    )

    try:
        device = next(model.parameters()).device
        best_or_last_model = best_model if best_model is not None else last_model
        _save_tensorboard_embeddings(tensorboard_dir, best_or_last_model.to(device), loaders[2], device)
    except Exception as e:
        print(f"Failed to save TensorBoard sentence embeddings: {e}")

    # Auto-detect and export BERT word embeddings
    try:
        best_or_last_model = best_model if best_model is not None else last_model
        if hasattr(best_or_last_model, "bert"):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            vocab = tokenizer.get_vocab()
            tokens = [token for token, idx in sorted(vocab.items(), key=lambda item: item[1])]
            word_embeddings = best_or_last_model.bert.embeddings.word_embeddings.weight
            _save_word_embeddings(tensorboard_dir, word_embeddings.detach().cpu(), tokens)
    except Exception as e:
        print(f"Failed to save BERT word embeddings: {e}")

    # Auto-detect LSTM embedding and export word embeddings if dataset provides tokens
    try:
        best_or_last_model = best_model if best_model is not None else last_model
        # get test dataset from loaders
        test_loader = loaders[2]
        raw_dataset = getattr(test_loader, "dataset", None)
        base_dataset = getattr(raw_dataset, "dataset", None) or raw_dataset
        tokens = getattr(base_dataset, "vocab_tokens", None)
        if tokens is not None and hasattr(best_or_last_model, "embedding"):
            word_embeddings = best_or_last_model.embedding.weight
            # Skip None placeholders (if any) and ensure tokens length matches embeddings
            cleaned_tokens = [t if t is not None else "<UNK>" for t in tokens]
            emb = word_embeddings.detach().cpu()
            if emb.size(0) == len(cleaned_tokens):
                _save_word_embeddings(tensorboard_dir, emb, cleaned_tokens)
            else:
                # If sizes don't match, try trimming/padding tokens
                min_len = min(emb.size(0), len(cleaned_tokens))
                _save_word_embeddings(tensorboard_dir, emb[:min_len], cleaned_tokens[:min_len])
    except Exception as e:
        print(f"Failed to save LSTM word embeddings: {e}")

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