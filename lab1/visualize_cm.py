import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_confusion_matrix(
    cm, 
    class_names,
    normalize: bool = True,
    save=True,
    save_dir="./output/",
    save_name="confusion_matrix.png",
    show=False,
    title="Confusion Matrix"
):
    fig, ax = plt.subplots()
    num_classes = max(4, len(class_names))
    fig.set_size_inches(num_classes, num_classes)
    fig.set_dpi(300)

    if normalize:
        row_sums = cm.sum(dim=1, keepdim=True).float()
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_display = cm.float() / row_sums
    else:
        cm_display = cm

    im = ax.imshow(cm_display)
    plt.colorbar(im)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (Normalized)" if normalize else " (Counts)"))

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j].item()

            if normalize:
                text = f"{round(val * 100)}%"
            else:
                text = f"{int(val)}"

            ax.text(
                j, i,
                text,
                ha="center",
                va="center",
                color=("black" if (val > 0.2 if normalize else val > cm.max() * 0.2) else "white")
            )

    plt.tight_layout()

    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, save_name))

    return fig

def load_cm(
    model_name,
    iteration_number
):
    output_file = f"./output/{model_name}/{iteration_number}/metadata.json"
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["confusion_matrix"]

def main():
    class_names = ["Negative", "Positive"]
    cm = load_cm("bert_yelp", 0)

    plot_confusion_matrix(torch.tensor(cm), class_names, save=False, show=True)


if __name__ == "__main__":
    main()