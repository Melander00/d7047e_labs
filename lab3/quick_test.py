import os
import torch
import random
import glob
from PIL import Image
from torchvision import transforms
from dataset import get_loaders

# Import our models
from Combinations.Base import BaseCaption
from Combinations.ResNet import ResNetCaption
from Combinations.Attention import AttentionCaption

DATA_DIR = "Data"

def run_inference_on_custom_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Vocabulary from the dataset
    try:
        _, dataset = get_loaders(DATA_DIR)
        vocab = dataset.vocab
        vocab_size = len(vocab)
        print(f"✅ Vocabulary loaded — {vocab_size} words")
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        print("   → Please make sure you are running this where 'Data/captions.txt' is accessible.")
        return

    # 2. Find any custom images in the current folder (lab3/)
    # It will search for any .jpg, .jpeg, or .png files that the user placed there.
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(ext))
    
    # Exclude dummy/test files if needed
    image_paths = [p for p in image_paths if os.path.basename(p) != "test.jpg"]

    if not image_paths:
        print("\n🔍 No custom test images found in the current folder!")
        print("💡 Place 3 or 4 of your own images (e.g. dog.jpg, cat.jpg) inside the 'lab3/' folder first, then run this script.")
        return

    print(f"🖼️  Found {len(image_paths)} image(s) to caption: {', '.join(image_paths)}")

    # 3. Load all three models
    models = {}
    model_configs = {
        "base": BaseCaption,
        "resnet": ResNetCaption,
        "attention": AttentionCaption
    }

    print("\nLoading trained models...")
    for model_name, model_class in model_configs.items():
        weights_path = f"SavedModels/{model_name}/best_comb.pth"
        if os.path.exists(weights_path):
            try:
                model = model_class(vocab_size).to(device)
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.eval()
                models[model_name] = model
                print(f"  ✅ {model_name.upper()} model loaded successfully.")
            except Exception as e:
                print(f"  ❌ Error loading {model_name} model: {e}")
        else:
            print(f"  ⚠️  No weights found for {model_name} at {weights_path} (Skipping).")

    if not models:
        print("❌ No trained models could be loaded. Please check your 'SavedModels/' directory.")
        return

    # 4. Prepare Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 5. Generate captions and compare
    print("\n" + "=" * 65)
    print("🔮  QUALITATIVE EVALUATION — SIDE-BY-SIDE CAPTION COMPARISON  🔮")
    print("=" * 65)

    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}] 🖼️  Image: {os.path.basename(path)}")
        print("-" * 65)
        
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                for model_name, model in models.items():
                    caption = model.generate_caption(image_tensor, vocab)
                    print(f"   💬 {model_name.upper():<10}: {caption}")
        except Exception as e:
            print(f"   ❌ Error processing image: {e}")
            
    print("\n" + "=" * 65)

if __name__ == "__main__":
    run_inference_on_custom_images()
