import json
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from visualize_cm import plot_confusion_matrix


def load_metadata_file(model_name, iteration_number):
    file_path = f"./output/{model_name}/{iteration_number}/metadata.json"
    if not os.path.isfile(file_path):
        raise RuntimeError("Model and/or iteration does not exist on disk")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def generate_csv_line(metadata):

    model_name = metadata['model_name']
    num_epochs = metadata['num_epochs']
    training_time = metadata['training_time']
    best_val_loss = metadata['best_val_loss']
    test_loss = metadata['test_loss']
    test_accuracy = metadata['test_accuracy']

    cm = np.array(metadata["confusion_matrix"])

    tn, fp = cm[0]
    fn, tp = cm[1]

    # (tn, tp) = (tp, tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    f1_macro = 1/2 * ( 2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp) )

    return f"{model_name};{num_epochs};{training_time};{best_val_loss};{test_loss};{test_accuracy};{f1};{f1_macro}"

def calc_training_time(metadata):
    return metadata['training_time']

def main():

    csv_entries = []
    csv_headers = "Model Name;Epochs;Training Time;Best Validation Loss;Test Loss;Test Accuracy;F1;F1 Macro"
    csv_entries.append(csv_headers)

    models = {
        "ann_amazon_1k": 1, 
        "ann_amazon_25k": 0, 
        "ann_yelp": 0,
        "lstm_amazon_1k": 1, 
        "lstm_amazon_25k": 1, 
        "lstm_yelp": 0,
        "bert_amazon_1k": 0, 
        "bert_amazon_25k": 0, 
        "bert_yelp": 0,
        "distilbert_amazon_1k": 0, 
        "distilbert_amazon_25k": 0, 
        "distilbert_yelp": 0,
    }

    total_time = 0

    for name in models:
        i = models[name]
        metadata = load_metadata_file(name, i)
        total_time += calc_training_time(metadata)
        csv = generate_csv_line(metadata)
        csv_entries.append(csv)

    print(total_time)

    csv_file = "metadata.csv"
    with open(csv_file, "w", newline="\n") as f:
        f.write("\n".join(csv_entries))
    
    print("Created CSV file")






if __name__ == "__main__":
    main()