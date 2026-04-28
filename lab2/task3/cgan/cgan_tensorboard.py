"""
TensorBoard writer for cGAN (Task 3)

Usage:
    python -m lab2.task3.cgan.tensorboard_writer
Then:
    tensorboard --logdir=./runs/cgan
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def load_metadata_file(model_name, version):
    file_path = f"./output/{model_name}/{version}/metadata.json"
    if not os.path.isfile(file_path):
        raise RuntimeError(f"Metadata not found at: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def write_losses(writer: SummaryWriter, metadata):
    """Write G and D loss curves per epoch."""
    for i in range(metadata["num_epochs"]):
        writer.add_scalar("Loss/G", metadata["g_losses"][i], i)
        writer.add_scalar("Loss/D", metadata["d_losses"][i], i)
        # Also log both on the same chart for easy comparison
        writer.add_scalars("Loss/GD", {
            "Generator": metadata["g_losses"][i],
            "Discriminator": metadata["d_losses"][i],
        }, i)


def write_hparams(writer: SummaryWriter, metadata):
    """Write hyperparameters and final metrics to the hparams tab."""
    hparams = {
        "lr_g": metadata["lr_g"],
        "lr_d": metadata["lr_d"],
        "batch_size": metadata["batch_size"],
        "z_dim": metadata["z_dim"],
        "h_dim_g": metadata["h_dim_g"],
        "h_dim_d": metadata["h_dim_d"],
        "num_epochs": metadata["num_epochs"],
    }

    metrics = {
        "hparam/best_g_loss": metadata["best_g_loss"],
        "hparam/final_g_loss": metadata["g_losses"][-1],
        "hparam/final_d_loss": metadata["d_losses"][-1],
        "hparam/training_time": metadata["training_time"],
    }

    writer.add_hparams(hparams, metrics)


def write_images(writer: SummaryWriter, model_name, version, step_every=1):
    """Write per-epoch sample images to TensorBoard."""
    img_dir = f"./output/{model_name}/{version}/img"

    if not os.path.isdir(img_dir):
        print(f"Warning: image directory not found at {img_dir} — skipping images.")
        return

    files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for file in files:
        epoch = int(os.path.splitext(file)[0])

        if epoch % step_every != 0:
            continue

        img_path = os.path.join(img_dir, file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)

        writer.add_image("Generated Samples", img_tensor, global_step=epoch)


def run_writer(model_name: str, version: str):
    metadata = load_metadata_file(model_name, version)

    log_dir = f"./runs/{model_name}/{version}"
    writer = SummaryWriter(log_dir)

    write_losses(writer, metadata)
    write_hparams(writer, metadata)
    write_images(writer, model_name, version, step_every=1)

    writer.close()
    print(f"TensorBoard logs written to: {log_dir}")
    print(f"Run: tensorboard --logdir=./runs/{model_name}")


def main():
    model_name = "cgan"

    for version in ["5", "10", "50", "100"]:
        output_path = f"./output/{model_name}/{version}/metadata.json"
        if os.path.isfile(output_path):
            print(f"Writing version {version}...")
            run_writer(model_name, version)
        else:
            print(f"Skipping version {version} — not trained yet.")


if __name__ == "__main__":
    main()