import torch
import torch.nn as nn
from dataset.loader import (load_amazon_simple, load_yelp_simple,
                            prepare_amazon_loaders, prepare_yelp_loaders)
from tqdm import tqdm
from training.training import develop_model
from transformers import AutoModel, AutoTokenizer


# Model Definition
class DistilBERTClassifier(nn.Module):
    def __init__(self, feature_extraction: bool = False) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        if feature_extraction:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # typically 768
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 2)  # binary classification

    def forward(self, x):
        # x shape: (batch, 2, seq_len)
        input_ids = x[:, 0, :]
        attention_mask = x[:, 1, :]

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        out = self.dropout(cls_output)
        out = self.fc(out)
        return out


# General Function to Execute Model Training
def exec_model(
    simple_dataset,
    loaders_fn,
    learning_rate=2e-5,
    model_name="distilbert",
    feature_extraction=False,
    iteration_number=0,
    num_epochs=10,
):
    # Compute class weights
    labels = {0: 0, 1: 0}
    for t, l in tqdm(simple_dataset, desc="Calculating label imbalance", leave=False):
        labels[l.item()] += 1
    # print("Label distribution:", labels)

    total = labels[0] + labels[1]

    loaders, dataset = loaders_fn()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBERTClassifier(feature_extraction=feature_extraction).to(device)

    # loss_weights = torch.tensor(
    #     [1.0, labels.get(0, 0) / labels.get(1, 1e-6)]
    # ).to(device)
    loss_weights = torch.tensor([total / labels[0], total / labels[1]]).to(device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    develop_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        model_name=model_name,
        num_epochs=num_epochs,
        iteration_number=iteration_number,
    )


# Amazon (1K and 25K) dataset
def exec_distilbert_amazon(
    use_25k_set=True,
    model_name="distilbert_amazon",
    iteration_number=0,
    num_epochs=10,
    max_len=128,
    batch_size=16,
    learning_rate=2e-5,
    feature_extraction=False,
):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    simple_dataset = load_amazon_simple(use_25k_set=use_25k_set)

    def loaders_fn():
        return prepare_amazon_loaders(
            use_25k_set=use_25k_set,
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_len=max_len,
        )

    exec_model(
        simple_dataset=simple_dataset,
        loaders_fn=loaders_fn,
        learning_rate=learning_rate,
        model_name=model_name,
        iteration_number=iteration_number,
        num_epochs=num_epochs,
        feature_extraction=feature_extraction,
    )


# Yelp dataset
def exec_distilbert_yelp(
    entries=100000,
    model_name="distilbert_yelp",
    iteration_number=0,
    num_epochs=10,
    max_len=128,
    batch_size=16,
    learning_rate=2e-5,
    feature_extraction=False,
):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    simple_dataset = load_yelp_simple(entries=entries)

    def loaders_fn():
        return prepare_yelp_loaders(
            entries=entries,
            batch_size=batch_size,
            tokenizer=tokenizer,
            max_len=max_len,
        )

    exec_model(
        simple_dataset=simple_dataset,
        loaders_fn=loaders_fn,
        learning_rate=learning_rate,
        model_name=model_name,
        iteration_number=iteration_number,
        num_epochs=num_epochs,
        feature_extraction=feature_extraction,
    )

# Main Training Loop
def main():
    max_len = 128
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 5
    feature_extraction = False

    # Amazon 25K
    exec_distilbert_amazon(
        use_25k_set=True,
        model_name="distilbert_amazon_25k",
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_len=max_len,
        iteration_number=0,
        num_epochs=num_epochs,
        feature_extraction=feature_extraction,
    )

    # Amazon 1K
    #exec_distilbert_amazon(
    #    use_25k_set=False,
    #    model_name="distilbert_amazon_1k",
    #    batch_size=batch_size,
    #    learning_rate=learning_rate,
    #    max_len=max_len,
    #    iteration_number=0,
    #    num_epochs=num_epochs,
    #    feature_extraction=feature_extraction,
    #)

    # Yelp (binary)
    # exec_distilbert_yelp(
    #    entries=100000,
    #    model_name="distilbert_yelp",
    #    batch_size=batch_size,
    #    learning_rate=learning_rate,
    #    max_len=max_len,
    #    iteration_number=0,
    #    num_epochs=num_epochs,
    #    feature_extraction=feature_extraction,
    #)


if __name__ == "__main__":
    main()