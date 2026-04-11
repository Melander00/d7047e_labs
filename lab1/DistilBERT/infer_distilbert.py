import os
import sys

import torch
from transformers import AutoTokenizer, logging
from DistilBERT.DistilBERT import DistilBERTClassifier  # adjust path if needed

logging.set_verbosity_error()
logging.disable_progress_bar()


def load_model_state(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/best_model.pt"
    if not os.path.isfile(file_path):
        raise RuntimeError(f"Model not found: {file_path}")

    return torch.load(file_path)["model_state"]


def start_inference(model_name, iteration_number="0"):

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load model architecture
    model = DistilBERTClassifier()

    # Load saved weights
    state_dict = load_model_state(model_name=model_name, iteration_number=iteration_number)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("=" * 6, "DistilBERT loaded. Start writing messages!", "=" * 6)

    while True:
        text = input("> ")

        if text.lower() in ["quit", "q", "exit", "e"]:
            print("Quitting...")
            break

        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Shape: (batch=1, 2, seq_len)
        inputs = torch.stack([input_ids, attention_mask], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(inputs)

        pred_class = torch.argmax(prediction, dim=1).item()
        confidence = torch.softmax(prediction, dim=1).squeeze(0)[pred_class]

        sentiment = "Negative >:(" if pred_class == 0 else "Positive!"
        print(f"Predicted: {sentiment} with {confidence:.2%} confidence")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError(
            "Usage: python infer_distilbert.py MODEL_NAME ITERATION_NUMBER\n"
            "Example: python infer_distilbert.py distilbert_amazon_25k 0"
        )

    try:
        start_inference(sys.argv[1], sys.argv[2])
    except KeyboardInterrupt:
        print("\nQuitting...")