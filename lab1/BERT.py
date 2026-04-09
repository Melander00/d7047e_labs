import torch
import torch.nn as nn
from dataset.loader import (load_amazon_simple, load_yelp_simple,
                            prepare_amazon_loaders, prepare_yelp_loaders)
from training.training import develop_model
from transformers import AutoModel, AutoTokenizer


class BERTClassifier(nn.Module):
    def __init__(self, feature_extraction = False) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        if feature_extraction:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, x):
        # x shape: (batch, 2, seq_len)
        input_ids = x[:, 0, :]
        attention_mask = x[:, 1, :]

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token
        cls_output = outputs.last_hidden_state[:, 0, :]

        out = self.dropout(cls_output)
        out = self.fc(out)

        return out










def exec_model(
    simple_dataset,
    loaders_fn,
    learning_rate=2e-5,
    model_name = "bert",
    feature_extraction = False,
    iteration_number=0,
    num_epochs = 10
):
    labels = {"0": 0, "1": 0}
    for t, l in simple_dataset:
        labels[str(l.item())] += 1
    print(labels)
    
    loaders, dataset = loaders_fn()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BERTClassifier(feature_extraction=feature_extraction).to(device)

    loss_weights = torch.tensor([1.0, labels.get("0", 0) / labels.get("1", 1e-6)]).to(device)

    criterion=nn.CrossEntropyLoss(weight=loss_weights)
    optim=torch.optim.Adam(model.parameters(),lr=learning_rate)

    develop_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optim,
        model_name=model_name,
        num_epochs=num_epochs,
        iteration_number=iteration_number
    )





def exec_bert_amazon(
        use_25k_set = True, 
        model_name = "bert_amazon", 
        iteration_number = 0,
        num_epochs = 10,
        max_len = 128,
        batch_size = 16,
        learning_rate = 2e-5,
        feature_extraction = False,
    ):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        feature_extraction=feature_extraction
    )




def exec_bert_yelp(
        entries = 100000,         
        model_name = "bert_yelp", 
        iteration_number = 0,
        num_epochs = 10,
        max_len = 128,
        batch_size = 16,
        learning_rate = 2e-5,
        feature_extraction = False,
    ):

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        feature_extraction=feature_extraction
    )


def main():
    max_len = 128
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 3  # Start with 3 epochs for a quick comparable result
    feature_extraction = False

    # Run 25K Amazon first — same dataset as ANN and LSTM for fair comparison
    exec_bert_amazon(
        use_25k_set=True,
        model_name="bert_amazon_25k",
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_len=max_len,
        iteration_number=0,
        num_epochs=num_epochs,
        feature_extraction=feature_extraction,
    )



if __name__ == "__main__":
    main()