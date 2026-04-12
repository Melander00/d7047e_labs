import re

import torch
import torch.nn as nn
from dataset.loader import (load_amazon_simple, load_yelp_simple,
                            prepare_amazon_loaders, prepare_yelp_loaders)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from training.training import develop_model


class simpleANN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # define layers here later
        self.seq=nn.Sequential (
            nn.Linear(vocab_size,64),
            
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            
           
            nn.Linear(64, 2),
        )
        

    def forward(self, x):
        return self.seq(x)
        # define forward pass later
        

def build_idf(simple_dataset, preprocess_fn):
    texts = []

    for text, _ in tqdm(simple_dataset, desc="Building TF-IDF", leave=False):
        text = preprocess_fn(text)
        texts.append(text)

    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=50000,
        max_df=0.5,
        use_idf=True,
        norm='l2'
    )

    vectorizer.fit(texts)

    return vectorizer

STOPWORDS = set(stopwords.words("english"))
def text_handler(text):
    text = text.lower()
    text = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', text)  # remove emails
    text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', text)  # remove IPs
    text = re.sub(r'[^\w\s]', '', text)  # remove special characters
    text = re.sub(r'\d', '', text)  # remove numbers

    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)


def exec_model(
    simple_dataset,
    loaders_fn,
    learning_rate=1e-3,
    model_name="lstm",
    iteration_number=0,
    num_epochs=10
):
    labels = {'0': 0, '1': 0}
    for t, l in tqdm(simple_dataset, desc="Calculating label imbalance", leave=False):
        labels[str(l.item())] += 1

    total = labels["0"] + labels["1"]

    vectorizer = build_idf(simple_dataset, text_handler)

    def preprocess(text):
        text = text_handler(text)
        vec = vectorizer.transform([text]).todense()
        vec = torch.tensor(vec, dtype=torch.float32).squeeze(0)
        return vec

    loaders, dataset = loaders_fn(preprocess)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = simpleANN(len(vectorizer.vocabulary_)).to(device)

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

def exec_ann_amazon(
    use_25k_set = True,
    model_name = "ann_amazon", 
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
    


def exec_ann_yelp(
    entries = int(1e6),
    model_name = "ann_yelp", 
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
    exec_ann_yelp(
        entries=int(1e5),
        model_name="ann_yelp",
        iteration_number=0,
        batch_size=128,
        learning_rate=5e-5,
        num_epochs=10,
    )