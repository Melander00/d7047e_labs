import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def load_metadata_file(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/metadata.json"
    if not os.path.isfile(file_path):
        raise RuntimeError("Model and/or iteration does not exist on disk")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data
        
def write_sd_losses(writer: SummaryWriter, metadata):
    epochs = metadata["num_epochs"]

    for i in range(epochs):
        writer.add_scalar("Loss", metadata["losses"][i], i)
        
def write_sd_lr(writer: SummaryWriter, metadata):
    epochs = metadata["num_epochs"]

    for i in range(epochs):
        writer.add_scalar("Learning rate", metadata["learning_rates"][i], i)
    
def write_sd_hparams(writer: SummaryWriter, metadata):
    hparams = {
        "lr": metadata["learning_rate"],
        "batch_size": metadata["batch_size"],
        "num_epochs": metadata["num_epochs"],
    }

    metrics = {
        "hparam/best_loss": metadata["best_loss"],
        "hparam/final_loss": metadata["losses"][-1],
    }

    writer.add_hparams(hparams, metrics)

def write_images(writer, model_name, version, step_every=10):
    img_dir = f"./output/{model_name}/{version}/img"

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Image directory does not exist: {img_dir}")

    files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(".png")
    ], key=lambda x: int(os.path.splitext(x)[0]))

    for file in files:
        epoch = int(os.path.splitext(file)[0])

        if epoch % step_every != 0:
            continue

        img_path = os.path.join(img_dir, file)

        # load image
        img = Image.open(img_path).convert("RGB")

        # convert to tensor (C, H, W)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)

        # log to tensorboard
        writer.add_image("Generated Samples", img_tensor, global_step=epoch)
    
def main():
    model_name = "unet_transformer"
    version = "1"

    metadata = load_metadata_file(model_name, version)

    log_dir = f"./runs/{model_name}/{version}"

    writer = SummaryWriter(log_dir)

    write_sd_losses(writer, metadata)
    write_sd_lr(writer, metadata)
    write_sd_hparams(writer, metadata)

    write_images(writer, model_name, version, step_every=1)

    writer.close()

if __name__ == "__main__":
    main()