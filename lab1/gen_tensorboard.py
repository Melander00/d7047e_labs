import json
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from visualize_cm import plot_confusion_matrix


def load_metadata_file(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/metadata.json"
    if not os.path.isfile(file_path):
        raise RuntimeError("Model and/or iteration does not exist on disk")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_losses_and_accuracies(metadata, writer: SummaryWriter):
    epochs = metadata['num_epochs']

    for i in range(epochs):
        writer.add_scalar("Loss/Train", metadata['train_loss'][i], i)
        writer.add_scalar("Loss/Val", metadata['val_loss'][i], i)
        writer.add_scalar("Accuracy/Train", metadata['train_accuracy'][i], i)
        writer.add_scalar("Accuracy/Val", metadata['val_accuracy'][i], i)

def write_params(metadata, writer: SummaryWriter):
    cm = np.array(metadata["confusion_matrix"])
    tn, fp = cm[0]
    fn, tp = cm[1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    hparams = {
        "model": metadata["model_name"],
        "epochs": metadata["num_epochs"]
    }
    metrics = {
        "hparam/test_accuracy": metadata["test_accuracy"],
        "hparam/test_loss": metadata["test_loss"],
        "hparam/f1": f1,
        "hparam/precision": precision,
        "hparam/recall": recall,
        "hparam/training_time": metadata["training_time"]
    }

    writer.add_hparams(hparams, metrics)
  

def write_singulars(metadata, writer: SummaryWriter):

    writer.add_scalar("Accuracy/Test", metadata["test_accuracy"])
    writer.add_scalar("Loss/Test", metadata["test_loss"])
    writer.add_scalar("Score/Time", metadata["training_time"])

def write_scores(metadata, writer: SummaryWriter):
    cm = np.array(metadata["confusion_matrix"])
    tn, fp = cm[0]
    fn, tp = cm[1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    writer.add_scalar("Score/F1", f1)
    writer.add_scalar("Score/Precision", precision)
    writer.add_scalar("Score/Recall", recall)

def write_cm(metadata, writer: SummaryWriter):
    cm = torch.tensor(metadata["confusion_matrix"])
    class_names = ["Negative", "Positive"]  # adjust as needed

    fig_counts = plot_confusion_matrix(
        cm,
        class_names,
        save=False,
        show=False,
        normalize=False
    )

    fig_normalized = plot_confusion_matrix(
        cm,
        class_names,
        save=False,
        show=False,
        normalize=True
    )

    writer.add_figure("CM/Counts", fig_counts)
    writer.add_figure("CM/Normalized", fig_normalized)

def generate_tensorboard_files(model_name, iteration_number):
    metadata = load_metadata_file(model_name, iteration_number)

    log_dir = f"./runs/{model_name}/{iteration_number}"

    writer = SummaryWriter(log_dir)

    write_losses_and_accuracies(metadata, writer)
    write_cm(metadata, writer) 

    write_scores(metadata, writer)
    write_singulars(metadata, writer)

    # write_params(metadata, writer)
    writer.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("Not enough arguments. Usage: python gen_tensorboard.py MODEL_NAME ITERATION_NUMBER")
    try:
        generate_tensorboard_files(sys.argv[1], sys.argv[2])
    except KeyboardInterrupt:
        print("\nQuitting....")