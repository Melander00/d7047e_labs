"""
compare_models.py
─────────────────
Run this after training all three models to generate a full comparison report.

Usage (from the d7047e_labs/ directory):
    python3 lab1/compare_models.py

Outputs:
    lab1/comparison/summary_table.txt
    lab1/comparison/training_curves.png
    lab1/comparison/confusion_matrices.png
    lab1/comparison/full_report.md
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ─── Model registry ───────────────────────────────────────────────────────────
# Maps a display name to the model_name and iteration used during training.
MODELS = {
    "Simple ANN": ("simple_ann_amazon25k", 0),
    "LSTM":       ("lstm_amazon25k",        0),
    "BERT":       ("bert_amazon_25k",       0),
}

OUTPUT_DIR = "./lab1/comparison"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_metadata(model_name: str, iteration: int) -> dict | None:
    path = f"./output/{model_name}/{iteration}/metadata.json"
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_metadata() -> dict:
    print("Loading metadata files...")
    all_meta = {}
    for display_name, (model_name, iteration) in MODELS.items():
        meta = load_metadata(model_name, iteration)
        if meta:
            all_meta[display_name] = meta
            print(f"  [OK] {display_name}")
        else:
            print(f"  [SKIP] {display_name} — train this model first")
    return all_meta


# ─── Plot 1: Training Curves ──────────────────────────────────────────────────

def plot_training_curves(all_meta: dict, save_path: str):
    colors = {"Simple ANN": "#e74c3c", "LSTM": "#3498db", "BERT": "#2ecc71"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — All Models (Amazon 25K)", fontsize=14, fontweight="bold")

    for name, meta in all_meta.items():
        epochs = range(1, len(meta["train_loss"]) + 1)
        c = colors.get(name, "gray")
        axes[0].plot(epochs, meta["train_loss"], linestyle="--", color=c, alpha=0.6, label=f"{name} (train)")
        axes[0].plot(epochs, meta["val_loss"],   linestyle="-",  color=c, linewidth=2, label=f"{name} (val)")
        axes[1].plot(epochs, meta["train_accuracy"], linestyle="--", color=c, alpha=0.6)
        axes[1].plot(epochs, meta["val_accuracy"],   linestyle="-",  color=c, linewidth=2, label=name)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Plot 2: Confusion Matrices ───────────────────────────────────────────────

def plot_confusion_matrices(all_meta: dict, save_path: str):
    n = len(all_meta)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

    class_names = ["Negative", "Positive"]

    for ax, (name, meta) in zip(axes, all_meta.items()):
        cm = torch.tensor(meta["confusion_matrix"]).float()
        row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)
        cm_norm  = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        test_acc = meta.get("test_accuracy", "N/A")
        acc_str  = f"{test_acc:.2%}" if isinstance(test_acc, float) else test_acc
        ax.set_title(f"{name}\nTest Acc: {acc_str}", fontweight="bold")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                val = cm_norm[i, j].item()
                ax.text(j, i, f"{val:.0%}",
                        ha="center", va="center",
                        color="white" if val > 0.5 else "black",
                        fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Summary Table ────────────────────────────────────────────────────────────

def print_and_save_summary(all_meta: dict, save_path: str):
    header = f"{'Model':<18} {'Best Val Acc':>12} {'Test Acc':>10} {'Test Loss':>10} {'Train Time':>12} {'Epochs':>7} {'Params':>12}"
    divider = "─" * len(header)

    lines = [divider, header, divider]

    # Static param counts (from our training logs)
    param_counts = {
        "Simple ANN": "1,280,322",
        "LSTM":       "  627,714",   # updated with new arch (64-emb, 128-hidden, 2-layer)
        "BERT":       "109,483,778",
    }

    for name, meta in all_meta.items():
        best_val = max(meta["val_accuracy"])
        test_acc = meta.get("test_accuracy", float("nan"))
        test_loss = meta.get("test_loss", float("nan"))
        train_time = meta.get("training_time", 0)
        epochs = meta.get("num_epochs", "?")
        params = param_counts.get(name, "?")

        val_str  = f"{best_val:.2%}"
        test_str = f"{test_acc:.2%}" if isinstance(test_acc, float) else "N/A"
        tl_str   = f"{test_loss:.4f}" if isinstance(test_loss, float) else "N/A"
        time_str = f"{train_time:.1f}s" if train_time < 120 else f"{train_time/60:.1f}m"

        lines.append(f"{name:<18} {val_str:>12} {test_str:>10} {tl_str:>10} {time_str:>12} {epochs:>7} {params:>12}")

    lines.append(divider)
    table = "\n".join(lines)

    print("\n" + table)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"  Saved: {save_path}")
    return table


# ─── Markdown Report ─────────────────────────────────────────────────────────

def generate_markdown_report(all_meta: dict, table: str, save_path: str):
    lines = []
    lines.append("# LAB 1 — Model Comparison Report\n")
    lines.append("> **Dataset**: Amazon Product Reviews (25K samples)\n")
    lines.append("> **Split**: 80% train / 10% val / 10% test (same for all models)\n\n")

    lines.append("## 📊 Performance Summary\n")
    lines.append("```\n" + table + "\n```\n")

    lines.append("## 📈 Training Curves\n")
    lines.append("![Training Curves](./training_curves.png)\n\n")

    lines.append("## 🔲 Confusion Matrices\n")
    lines.append("![Confusion Matrices](./confusion_matrices.png)\n\n")

    lines.append("## 🔍 Analysis\n\n")

    # Best model
    best_name = max(all_meta, key=lambda n: all_meta[n].get("test_accuracy", 0))
    best_acc  = all_meta[best_name].get("test_accuracy", 0)
    lines.append(f"### Winner: {best_name} ({best_acc:.2%} test accuracy)\n\n")

    lines.append("### Task 1.3 Answers\n\n")
    lines.append("**Q: Compare performance and explain when to prefer each model.**\n\n")
    lines.append(
        "- **Simple ANN**: Fastest training (~seconds). Best when you need quick iteration "
        "and the dataset is small-medium. Bag-of-words approach loses word order.\n"
        "- **LSTM**: Slower training (~1-2 min). Better suited when word order and "
        "sequential context matter. More robust than ANN on longer texts.\n"
        "- **BERT**: Slowest training (~minutes per epoch). Best accuracy due to pre-trained "
        "knowledge. Prefer when accuracy is critical and compute is available.\n\n"
    )

    lines.append("**Q: How did complexity, accuracy, and efficiency differ?**\n\n")
    lines.append(
        "| Aspect | Simple ANN | LSTM | BERT |\n"
        "|--------|-----------|------|------|\n"
        "| Complexity | Low | Medium | Very High |\n"
        "| Accuracy | Medium | Medium | High |\n"
        "| Training Speed | Fast | Medium | Slow |\n"
        "| Memory | Low | Medium | High |\n\n"
    )

    lines.append("**Q: Insights on data amount, embeddings, and architecture.**\n\n")
    lines.append(
        "- **Data**: ANN's TF-IDF is data-efficient — good results with limited data. "
        "BERT benefits most from larger datasets due to fine-tuning dynamics.\n"
        "- **Embeddings**: TF-IDF (ANN) is static and sparse. LSTM learns 64-dim dense "
        "embeddings. BERT uses 768-dim contextual embeddings pre-trained on Wikipedia.\n"
        "- **Architecture**: ANN has no memory (bag-of-words). LSTM remembers sequence "
        "order. BERT sees all words simultaneously with bidirectional attention — "
        "the most powerful representation.\n"
    )

    report = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_meta = load_all_metadata()

    if not all_meta:
        print("\nNo metadata found. Train at least one model first.")
        return

    print(f"\nFound {len(all_meta)} trained model(s): {', '.join(all_meta.keys())}\n")

    print("Generating summary table...")
    table = print_and_save_summary(
        all_meta,
        save_path=os.path.join(OUTPUT_DIR, "summary_table.txt")
    )

    print("\nPlotting training curves...")
    plot_training_curves(
        all_meta,
        save_path=os.path.join(OUTPUT_DIR, "training_curves.png")
    )

    print("\nPlotting confusion matrices...")
    plot_confusion_matrices(
        all_meta,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    )

    print("\nGenerating markdown report...")
    generate_markdown_report(
        all_meta,
        table=table,
        save_path=os.path.join(OUTPUT_DIR, "full_report.md")
    )

    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
