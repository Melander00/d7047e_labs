import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "Best Validation Loss",
    "Test Loss",
    "Test Accuracy",
    "F1 Macro",
]

LOSS_COLUMNS = [
    "Best Validation Loss",
    "Test Loss",
]

COMPARISON_DIR = Path(__file__).resolve().parent


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")
    required = ["Model Name", "Epochs", "Training Time", *METRIC_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_loss_plot(df: pd.DataFrame, output_path: Path) -> None:
    # Sort so lower losses and higher accuracy stand out more clearly.
    by_accuracy = df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    labels = by_accuracy["Model Name"]

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle("Model Loss Comparison", fontsize=18, fontweight="bold")

    colors = {
        "Best Validation Loss": "#4C72B0",
        "Test Loss": "#DD8452",
        "Test Accuracy": "#55A868",
    }

    x = range(len(labels))
    width = 0.35

    for idx, metric in enumerate(LOSS_COLUMNS):
        values = by_accuracy[metric]
        offset = (idx - 0.5) * width
        positions = [i + offset for i in x]
        bars = ax.bar(positions, values, width=width, color=colors[metric], alpha=0.9, label=metric)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_accuracy_plot(df: pd.DataFrame, output_path: Path) -> None:
    by_accuracy = df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    labels = by_accuracy["Model Name"]
    accuracy_values = by_accuracy["Test Accuracy"]
    f1_macro_values = by_accuracy["F1 Macro"]

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle("Model Test Accuracy and F1 Macro Comparison", fontsize=18, fontweight="bold")

    x = range(len(labels))
    width = 0.35

    accuracy_positions = [i - width / 2 for i in x]
    f1_macro_positions = [i + width / 2 for i in x]

    accuracy_bars = ax.bar(
        accuracy_positions,
        accuracy_values,
        width=width,
        color="#55A868",
        alpha=0.9,
        label="Test Accuracy",
    )
    f1_macro_bars = ax.bar(
        f1_macro_positions,
        f1_macro_values,
        width=width,
        color="#C44E52",
        alpha=0.9,
        label="F1 Macro",
    )

    for bar, value in zip(accuracy_bars, accuracy_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    for bar, value in zip(f1_macro_bars, f1_macro_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_runtime_plot_for_epoch(df: pd.DataFrame, epoch_count: int, output_path: Path) -> None:
    filtered = df[df["Epochs"] == epoch_count].copy()
    if filtered.empty:
        raise ValueError(f"No models found with {epoch_count} epochs")

    by_time = filtered.sort_values("Training Time", ascending=False).reset_index(drop=True)
    labels = by_time["Model Name"]
    time_values = by_time["Training Time"]

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.suptitle(f"Model Training Time Comparison ({epoch_count} Epochs)", fontsize=18, fontweight="bold")

    bars = ax.bar(labels, time_values, color="#8172B3", alpha=0.9, label=f"Training Time ({epoch_count} epochs)")

    for bar, time_value in zip(bars, time_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{time_value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            rotation=0,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Time (seconds)")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate histogram-style plots for losses, accuracy, training time, and epochs."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=COMPARISON_DIR / "metadata.csv",
        help="Input CSV path (default: <this file's folder>/metadata.csv)",
    )
    parser.add_argument(
        "--out-loss",
        type=Path,
        default=COMPARISON_DIR / "loss_histogram.png",
        help="Loss output image path (default: <this file's folder>/loss_histogram.png)",
    )
    parser.add_argument(
        "--out-accuracy",
        type=Path,
        default=COMPARISON_DIR / "accuracy_F1_histogram.png",
        help="Accuracy output image path (default: <this file's folder>/accuracy_F1_histogram.png)",
    )
    parser.add_argument(
        "--out-runtime-10",
        type=Path,
        default=COMPARISON_DIR / "runtime_10_epochs_histogram.png",
        help="Runtime output for 10-epoch models (default: <this file's folder>/runtime_10_epochs_histogram.png)",
    )
    parser.add_argument(
        "--out-runtime-50",
        type=Path,
        default=COMPARISON_DIR / "runtime_50_epochs_histogram.png",
        help="Runtime output for 50-epoch models (default: <this file's folder>/runtime_50_epochs_histogram.png)",
    )
    args = parser.parse_args()

    df = load_metadata(args.csv)
    build_loss_plot(df, args.out_loss)
    build_accuracy_plot(df, args.out_accuracy)
    build_runtime_plot_for_epoch(df, 10, args.out_runtime_10)
    build_runtime_plot_for_epoch(df, 50, args.out_runtime_50)
    print(f"Saved loss histogram plot to: {args.out_loss}")
    print(f"Saved accuracy histogram plot to: {args.out_accuracy}")
    print(f"Saved runtime histogram plot (10 epochs) to: {args.out_runtime_10}")
    print(f"Saved runtime histogram plot (50 epochs) to: {args.out_runtime_50}")


if __name__ == "__main__":
    main()
