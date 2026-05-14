import torch
from PIL import Image
from torchvision import transforms
from Models.CNN import CNN
from Models.RNN import CaptionRNN
from dataset import get_loaders
import os
import random

# ============================================================
# CONFIGURE THIS: Path to your Flickr8k data folder
# The folder must contain: Images/ subfolder + captions.txt
# Example: DATA_DIR = "/home/user/flickr8k"
# ============================================================
DATA_DIR = "Data"


def run_multiple_captions(image_paths, model_name="steve"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Vocabulary from the dataset
    try:
        _, dataset = get_loaders(DATA_DIR)
        vocab = dataset.vocab
        print(f"✅ Vocabulary loaded — {len(vocab)} words")
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        print(f"   → Make sure DATA_DIR is set correctly at the top of this file.")
        return

    # 2. Initialize and Load Models (Only once for all images)
    model_cnn = CNN().to(device)
    model_rnn = CaptionRNN(vocab_size=len(vocab)).to(device)

    try:
        model_cnn.load_state_dict(torch.load(
            f"SavedModels/{model_name}/{model_name}_CNN.pth", map_location=device))
        model_rnn.load_state_dict(torch.load(
            f"SavedModels/{model_name}/{model_name}_RNN.pth", map_location=device))
        print(f"✅ Weights loaded for '{model_name}' model (using {device.type.upper()})")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    model_cnn.eval()
    model_rnn.eval()

    # 3. Prepare Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("\n" + "=" * 50)
    print(f"   🔍 GENERATING CAPTIONS — Model: {model_name}")
    print("=" * 50)

    # 4. Loop through all images and generate captions
    for i, path in enumerate(image_paths, 1):
        if not os.path.exists(path):
            print(f"\n[{i}] ❌ File not found: {path}")
            continue

        try:
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model_cnn(image_tensor)
                caption = model_rnn.generate_caption(features, vocab=vocab)

            print(f"\n[{i}] 🖼️  {os.path.basename(path)}")
            print(f"     💬  {caption}")
        except Exception as e:
            print(f"\n[{i}] ❌ Error processing {path}: {e}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    # ============================================================
    # STEP 1: Choose your model
    # Options: "base", "resnet", "attention"
    # "base" → uses the "steve" saved weights
    # ============================================================
    model_type = "base"

    # ============================================================
    # STEP 2: Get 4 random images from the test split automatically
    # ============================================================
    my_images = []
    try:
        loaders, dataset = get_loaders(DATA_DIR)
        test_loader = loaders[2]  # 15% test split

        print("📥 Sampling 4 images from the test split...")
        test_indices = test_loader.dataset.indices
        sample_indices = random.sample(list(test_indices), 4)

        for idx in sample_indices:
            img_name = dataset.images[idx]
            full_path = os.path.join(DATA_DIR, "Images", img_name)
            my_images.append(full_path)

    except Exception as e:
        print(f"⚠️  Could not load test split: {e}")
        print("   → Make sure DATA_DIR is set correctly at the top of this file.")

    # ============================================================
    # STEP 3: Run captioning
    # ============================================================
    if my_images:
        folder = "steve" if model_type == "base" else model_type
        run_multiple_captions(my_images, model_name=folder)
    else:
        print("❌ No images found. Cannot run inference.")
