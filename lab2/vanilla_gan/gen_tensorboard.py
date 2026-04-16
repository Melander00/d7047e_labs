import json
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def load_metadata_file(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/metadata.json"
    if not os.path.isfile(file_path):
        raise RuntimeError("Model and/or iteration does not exist on disk")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_losses(writer: SummaryWriter, metadata):
    epochs = metadata["num_epochs"]

    for i in range(epochs):
        writer.add_scalar("Loss/D", metadata["d_losses"][i], i)
        writer.add_scalar("Loss/G", metadata["g_losses"][i], i)


def write_hparams(writer: SummaryWriter, metadata):
    hparams = {
        "lr": metadata["learning_rate"],
        "batch_size": metadata["batch_size"],
        "z_dim": metadata["z_dim"],
        "h_dim": metadata["h_dim"],
        "num_epochs": metadata["num_epochs"],
    }

    metrics = {
        "hparam/best_g_loss": metadata["best_g_loss"],
        "hparam/final_g_loss": metadata["g_losses"][-1],
        "hparam/final_d_loss": metadata["d_losses"][-1],
    }

    writer.add_hparams(hparams, metrics)


def main():
    model_name = "vanilla_gan"
    version = "2"

    metadata = load_metadata_file(model_name, version)

    log_dir = f"./runs/{model_name}/{version}"

    writer = SummaryWriter(log_dir)

    write_losses(writer, metadata)
    write_hparams(writer, metadata)

    writer.close()

if __name__ == "__main__":
    main()