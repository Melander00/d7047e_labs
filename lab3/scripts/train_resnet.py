from Combinations.ResNet import ResNetCaption
import torch
from dataset import get_loaders
from Combinations.Combination_Trainer import train_model


def main(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, dataset = get_loaders(data_dir)

    model = ResNetCaption(len(dataset.vocab)).to(device)

    train_model(
        model,
        "resnet",
        25,
        data_dir,
        output_dir="SavedModels"
    )


if __name__ == "__main__":
    main("Data")