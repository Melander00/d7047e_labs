import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset.loader import prepare_yelp_loaders
from torch.utils.data import DataLoader, TensorDataset
from Task1.task1_data import data_loading_code as loader
from training.training import run_training, run_model, run_test, develop_model
from Task1.Models.Basic_ANN import simpleANN
from Task1.Models.Lstm_model import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  SIMPLE ANN
# ─────────────────────────────────────────────────────────────────────────────

def main_simple_ann(save_board=True):
    # Cache uses v2 naming — 7-value format including test split
    CACHE_PATH    = "ANN_cached_data_v3.pt"
    text_with_data = "amazon_cells_labelled_LARGE_25K.txt"

    if os.path.exists(CACHE_PATH):
        print("Loading cached dataset...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels, vocab = \
            torch.load(CACHE_PATH, weights_only=False)
    else:
        print("Processing raw dataset and creating cache...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels, vocab = \
            loader.load_prep_data(text_with_data=text_with_data)
        torch.save(
            (train_data, train_labels, val_data, val_labels, test_data, test_labels, vocab),
            CACHE_PATH
        )
        print("Saved dataset to cache.")

    dataset_train = TensorDataset(train_data, train_labels)
    dataset_val   = TensorDataset(val_data,   val_labels)
    dataset_test  = TensorDataset(test_data,  test_labels)

    dataloader_training   = DataLoader(batch_size=64, shuffle=True,  dataset=dataset_train)
    dataloader_validation = DataLoader(batch_size=64, shuffle=False, dataset=dataset_val)
    dataloader_testing    = DataLoader(batch_size=64, shuffle=False, dataset=dataset_test)
    loaders = (dataloader_training, dataloader_validation, dataloader_testing)

    vocab_size = len(vocab)
    model      = simpleANN(vocab_size).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.CrossEntropyLoss()

    metadata = develop_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        model_name="simple_ann_amazon25k",
        num_epochs=10,
        iteration_number=0
    )

    if save_board:
        writer = SummaryWriter("runs_task1/ANN")
        for epoch in range(len(metadata['train_loss'])):
            writer.add_scalar("Loss/Train",      metadata['train_loss'][epoch],      epoch)
            writer.add_scalar("Loss/Val",        metadata['val_loss'][epoch],        epoch)
            writer.add_scalar("Accuracy/Train",  metadata['train_accuracy'][epoch],  epoch)
            writer.add_scalar("Accuracy/Val",    metadata['val_accuracy'][epoch],    epoch)
        writer.close()
        print("TensorBoard logs saved to runs_task1/ANN")


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM
# ─────────────────────────────────────────────────────────────────────────────

def main_LSTM(save_board=True):
    # Cache uses v2 naming — 7-value format including test split
    CACHE_PATH    = "lstm_cached_data_v3.pt"
    text_with_data = "amazon_cells_labelled_LARGE_25K.txt"

    if os.path.exists(CACHE_PATH):
        print("Loading cached dataset...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels, vocab_size = \
            torch.load(CACHE_PATH, weights_only=False)
    else:
        print("Processing raw dataset and creating cache...")
        train_data, train_labels, val_data, val_labels, test_data, test_labels, vocab_size = \
            loader.load_prep_data_lstm(text_with_data=text_with_data)
        torch.save(
            (train_data, train_labels, val_data, val_labels, test_data, test_labels, vocab_size),
            CACHE_PATH
        )
        print("Saved dataset to cache.")

    print("vocab_size =", vocab_size)

    dataset_train = TensorDataset(train_data, train_labels)
    dataset_val   = TensorDataset(val_data,   val_labels)
    dataset_test  = TensorDataset(test_data,  test_labels)

    dataloader_training   = DataLoader(batch_size=64, shuffle=True,  dataset=dataset_train)
    dataloader_validation = DataLoader(batch_size=64, shuffle=False, dataset=dataset_val)
    dataloader_testing    = DataLoader(batch_size=64, shuffle=False, dataset=dataset_test)
    loaders = (dataloader_training, dataloader_validation, dataloader_testing)

    model     = LSTM(vocab_size).to(device)

    # Calculate class weights to prevent majority-class bias
    count_pos = (train_labels == 1).sum().item()
    count_neg = (train_labels == 0).sum().item()
    loss_weights = torch.tensor([1.0, count_neg / count_pos]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Lowered LR
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    metadata = develop_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        model_name="lstm_amazon25k",
        num_epochs=10,
        iteration_number=0
    )

    if save_board:
        writer = SummaryWriter("runs_task1/LSTM")
        for epoch in range(len(metadata['train_loss'])):
            writer.add_scalar("Loss/Train",      metadata['train_loss'][epoch],      epoch)
            writer.add_scalar("Loss/Val",        metadata['val_loss'][epoch],        epoch)
            writer.add_scalar("Accuracy/Train",  metadata['train_accuracy'][epoch],  epoch)
            writer.add_scalar("Accuracy/Val",    metadata['val_accuracy'][epoch],    epoch)
        writer.close()
        print("TensorBoard logs saved to runs_task1/LSTM")


# ─────────────────────────────────────────────────────────────────────────────
#  LARGE DATASET (Yelp) — requires yelp_academic_dataset_review.json
# ─────────────────────────────────────────────────────────────────────────────

def main_bigdata(Model="LSTM"):
    (traindata, valdata, testdata), dataset = prepare_yelp_loaders(batch_size=64)

    # Build vocab size from training data
    max_id = 0
    for data, _ in traindata:
        max_id = max(max_id, data.max().item())
    vocab_size = max_id + 1
    print("Vocab size:", vocab_size)

    if Model == "LSTM":
        model = LSTM(vocab_size).to(device)
    elif Model == "ANN":
        model = simpleANN(vocab_size).to(device)

    loaders   = (traindata, valdata, testdata)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metadata = develop_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        model_name=f"{Model.lower()}_yelp",
        num_epochs=3,
        iteration_number=0
    )
    print("Training time:", metadata['training_time'])


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
#  Run both ANN and LSTM back-to-back for a full comparison run.
#  Comment out whichever you don't need.
# ─────────────────────────────────────────────────────────────────────────────

main_simple_ann()
main_LSTM()
#main_bigdata()
# tensorboard --logdir=runs_task1    ← run this in the terminal to view graphs