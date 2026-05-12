import torch
from PIL import Image
from torchvision import transforms
from Combinations.CaptionModel import CaptionModel
from dataset import get_loaders
import matplotlib.pyplot as plt

from Combinations.Attention import AttentionCaption
from Combinations.Base import BaseCaption
from Combinations.ResNet import ResNetCaption

def visualize_image(image_path, caption=""):
    image = Image.open(image_path).convert("RGB")

    plt.title(caption)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0) # type: ignore

    return image

def infer(image_path: str, model: CaptionModel, vocab, max_length = 20):

    device = next(iter(model.parameters())).device

    image = load_image(image_path).to(device)
    with torch.no_grad():
        caption = model.generate_caption(image, vocab, max_length)
    return caption


def main(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, dataset = get_loaders("data")

    vocab = dataset.vocab

    # model = AttentionCaption(len(vocab)).to(device)
    # model.load_state_dict(torch.load("SavedModels/attention/best_comb.pth"))

    model = BaseCaption(len(vocab)).to(device)
    model.load_state_dict(torch.load("SavedModels/base/best_comb.pth"))

    caption = infer(image_path, model, vocab, 20)
    visualize_image(image_path, caption)



if __name__ == "__main__":
    main("test.jpg")