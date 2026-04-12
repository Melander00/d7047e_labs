import re
from collections import Counter

import torch
import torch.nn as nn
from dataset.loader import (load_amazon_simple, load_yelp_simple,
                            prepare_amazon_loaders, prepare_yelp_loaders)
from nltk.corpus import stopwords
from tqdm import tqdm
from training.training import develop_model


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64,padding_idx=0)
        
        self.lstm=nn.LSTM(64,128,batch_first=True,num_layers=1)
        
        self.dropout = nn.Dropout(0.3)

        self.fc1=nn.Linear(128,2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)

        return out
    

class Vocab():
    def __init__(self) -> None:
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
    
    def build(self, dataset, min_freq = 1, text_handler = None):
        counter = Counter()
        for text, _ in tqdm(dataset, desc="Building vocab", leave=False, unit=" words"):
            if text_handler:
                text = " ".join(text_handler(text))
            for word in text.split():
                counter[word] += 1
        idx = 2
        for word, freq in counter.items():
            if freq >= min_freq:
                self.vocab[word] = idx
                idx += 1
    
    def word2idx(self, word: str):
        return self.vocab.get(word, self.vocab["<UNK>"])
    
    def get_pad(self):
        return self.vocab.get("<PAD>")

    def size(self):
        return len(self.vocab)

STOPWORDS = set(stopwords.words("english"))
def text_handler(text):
    text = text.lower()
    text = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', text)  # remove emails
    text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', text)  # remove IPs
    text = re.sub(r'[^\w\s]', '', text)  # remove special characters
    text = re.sub(r'\d', '', text)  # remove numbers

    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]
    return tokens


def exec_model(
    simple_dataset,
    loaders_fn,
    learning_rate=1e-3,
    model_name="lstm",
    iteration_number=0,
    num_epochs=10,
    max_len=128,
):
    vocab = Vocab()
    vocab.build(simple_dataset, text_handler=text_handler, min_freq=2)

    labels = {'0': 0, '1': 0}
    for t, l in tqdm(simple_dataset, desc="Calculating label imbalance", leave=False):
        labels[str(l.item())] += 1

    # print(f"Label imbalance: {labels}")

    total = labels["0"] + labels["1"]

    def preprocess(text):
        tokens = text_handler(text)

        indices = [vocab.word2idx(w) for w in tokens]
        # indices = [vocab.word2idx(word) for word in text.split()]

        indices = indices[:max_len]

        pad_len = max_len - len(indices)
        indices += [vocab.get_pad()] * pad_len

        return torch.tensor(indices)

    loaders, dataset = loaders_fn(preprocess)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = LSTM(vocab.size()).to(device)
    
    # loss_weights = torch.tensor([labels["0"] / total, labels["1"] / total]).to(device)
    loss_weights = torch.tensor([total / labels["0"], total / labels["1"]]).to(device)

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

def exec_lstm_amazon(
    use_25k_set = True,
    model_name = "lstm_amazon", 
    iteration_number = 0,
    num_epochs = 10,
    batch_size = 16,
    learning_rate = 1e-3,
):
    simple_dataset = load_amazon_simple(use_25k_set=use_25k_set)

    def loaders_fn(preprocessor):
        return prepare_amazon_loaders(
            use_25k_set=use_25k_set,
            batch_size=batch_size,
            text_preprocessing=preprocessor
        )
    
    exec_model(
        simple_dataset=simple_dataset,
        loaders_fn=loaders_fn,
        model_name=model_name,
        iteration_number=iteration_number,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    


def exec_lstm_yelp(
    entries = int(1e6),
    model_name = "lstm_yelp", 
    iteration_number = 0,
    num_epochs = 10,
    batch_size = 16,
    learning_rate = 1e-3,
):
    simple_dataset = load_yelp_simple(entries=entries)

    def loaders_fn(preprocessor):
        return prepare_yelp_loaders(
            entries=entries,
            batch_size=batch_size,
            text_preprocessing=preprocessor
        )
    
    exec_model(
        simple_dataset=simple_dataset,
        loaders_fn=loaders_fn,
        model_name=model_name,
        iteration_number=iteration_number,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )



if __name__ == "__main__":
    exec_lstm_yelp(
        entries=int(1e5),
        model_name="lstm_yelp",
        iteration_number=0,
        batch_size=128,
        learning_rate=1e-3,
        num_epochs=20,
    )