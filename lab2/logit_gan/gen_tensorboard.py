"""
gen_tensorboard.py — Task 2 (Logit GAN)

Reads existing metadata.json files from your 3 training runs
(5, 10, and 50 epochs) and writes TensorBoard log files.

NO re-training needed. Just run this script once on the server 
after training is already done.

Usage (from lab2/logit_gan/ directory):
    python gen_tensorboard.py

TensorBoard files will be saved to:
    ./runs/logit_gan/5/
    ./runs/logit_gan/10/
    ./runs/logit_gan/50/

To view in TensorBoard:
    tensorboard --logdir=./runs/logit_gan
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def load_metadata(model_name: str, version: str):
    """Load the metadata.json saved after training."""
    file_path = f"./output/{model_name}/{version}/metadata.json"
    if not os.path.isfile(file_path):
        raise RuntimeError(f"metadata.json not found at: {file_path}\n"
                           "Make sure you have run logit_main.py on the GPU server first.")
    with open(file_path, "r") as f:
        return json.load(f)


def write_losses(writer: SummaryWriter, metadata: dict):
    """Write G and D loss curves — one point per epoch."""
    for i, (g_loss, d_loss) in enumerate(zip(metadata["g_losses"], metadata["d_losses"])):
        writer.add_scalar("Loss/Generator",     g_loss, i)
        writer.add_scalar("Loss/Discriminator", d_loss, i)


def write_hparams(writer: SummaryWriter, metadata: dict):
    """Write hyperparameters so TensorBoard can compare runs side by side."""
    hparams = {
        "lr":         metadata["learning_rate"],
        "batch_size": metadata["batch_size"],
        "z_dim":      metadata["z_dim"],
        "h_dim":      metadata["h_dim"],
        "num_epochs": metadata["num_epochs"],
        "loss_fn":    metadata.get("loss_function", "BCEWithLogitsLoss"),
    }

    metrics = {
        "hparam/best_g_loss":  metadata["best_g_loss"],
        "hparam/final_g_loss": metadata["g_losses"][-1],
        "hparam/final_d_loss": metadata["d_losses"][-1],
    }

    writer.add_hparams(hparams, metrics)


def write_images(writer: SummaryWriter, model_name: str, version: str, step_every: int = 1):
    """Write generated images so you can scrub through them in TensorBoard."""
    img_dir = f"./output/{model_name}/{version}/img"

    if not os.path.isdir(img_dir):
        print(f"  [Warning] No image directory found at: {img_dir} — skipping images.")
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


def generate_tensorboard_files(model_name: str, version: str):
    """Main function: load metadata and write all TensorBoard data."""
    print(f"  Generating TensorBoard files for {model_name} / {version} epochs...")

    metadata = load_metadata(model_name, version)

    log_dir = f"./runs/{model_name}/{version}"
    writer = SummaryWriter(log_dir)

    write_losses(writer, metadata)
    write_hparams(writer, metadata)
    write_images(writer, model_name, version, step_every=1)

    writer.close()
    print(f"  Done! Logs saved to: {log_dir}")


def main():
    model_name = "logit_gan"
    versions = ["5", "10", "50"]   # The 3 runs you already trained

    print("=" * 50)
    print("Generating TensorBoard files for Task 2 (Logit GAN)")
    print("=" * 50)

    for version in versions:
        try:
            generate_tensorboard_files(model_name, version)
        except RuntimeError as e:
            print(f"  [Skipped] {e}")

    print()
    print("All done! To view in TensorBoard, run:")
    print("  tensorboard --logdir=./runs/logit_gan")
    print("=" * 50)


if __name__ == "__main__":
    main()
