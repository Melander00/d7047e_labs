import os
import sys

import torch
from bert.BERT import BERTClassifier
from transformers import AutoTokenizer, logging

logging.set_verbosity_error()
logging.disable_progress_bar()


def load_model_state(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/best_model.pt"
    if not os.path.isfile(file_path):
        raise RuntimeError("Model and/or iteration does not exist on disk")

    return torch.load(file_path)["model_state"]

def start_inference(model_name, iteration_number="0"):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = BERTClassifier()
    state_dict = load_model_state(model_name=model_name, iteration_number=iteration_number)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("")
    print("="*6,"BERT loaded. Start writing messages!", "="*6)

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

        inputs = torch.stack([input_ids, attention_mask], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(inputs)

        pred_class = torch.argmax(prediction, dim=1).item()

        prob = 0 if pred_class == 0 else 1

        confidence = torch.softmax(prediction, dim=1).squeeze(0)[prob]

        sentiment = "Negative >:(" if pred_class == 0 else "Positive!"
        color = "\033[91m" if pred_class == 0 else "\033[92m"
        reset = "\033[00m" 
        print(f"Predicted: {color}{sentiment}{reset} with {confidence:.2%} confidence")
        print("")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise RuntimeError("Not enough arguments. Usage: python infer_bert.py MODEL_NAME ITERATION_NUMBER")
    try:
        start_inference(sys.argv[1], sys.argv[2])
    except KeyboardInterrupt:
        print("\nQuitting....")