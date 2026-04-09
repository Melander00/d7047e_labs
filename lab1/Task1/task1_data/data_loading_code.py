import os

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)
    data['Sentence'] = data['Sentence'].replace(r'\d', '', regex=True)
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens] # No stopword removal, matches original 82% run
        df_.loc[len(df_)] = {
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent)
        }
    return df_

def load_prep_data(text_with_data="amazon_cells_labelled.txt"):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, text_with_data)
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    sentences = data['Sentence'].values.astype('U')
    labels = data['Class'].values.astype('int32')

    # Reverting to the simpler split used in the 82% run
    train_s, val_s, train_l, val_l = train_test_split(sentences, labels, test_size=0.10, random_state=0)
    test_s, test_l = val_s, val_l # Use val as test just to keep it running for now

    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=20000, max_df=0.5, use_idf=True, norm='l2')
    train_x = torch.from_numpy(np.array(word_vectorizer.fit_transform(train_s).todense())).float()
    val_x = torch.from_numpy(np.array(word_vectorizer.transform(val_s).todense())).float()
    test_x = val_x

    train_y = torch.from_numpy(np.array(train_l)).long()
    val_y = torch.from_numpy(np.array(val_l)).long()
    test_y = val_y

    return train_x, train_y, val_x, val_y, test_x, test_y, word_vectorizer.vocabulary_

def load_prep_data_lstm(text_with_data="amazon_cells_labelled.txt"):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, text_with_data)
    data = pd.read_csv(file_path, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    sentences = data['Sentence'].values.astype('U')
    labels = data['Class'].values.astype('int32')

    train_s, val_s, train_l, val_l = train_test_split(sentences, labels, test_size=0.10, random_state=0)
    test_s, test_l = val_s, val_l

    def tokenize(text):
        return text.lower().split()

    # Reverting to building vocab on the whole dataset to match the 82% performance
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in sentences:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vocab_size = len(vocab)
    max_len = 20 # Reverting to original length 20

    def encode_and_pad(text):
        ids = [vocab.get(w, vocab["<UNK>"]) for w in tokenize(text)]
        return ids[:max_len] + [0] * max(0, max_len - len(ids))

    train_x = torch.tensor([encode_and_pad(t) for t in train_s]).long()
    val_x = torch.tensor([encode_and_pad(t) for t in val_s]).long()
    test_x = val_x

    train_y = torch.tensor(train_l).long()
    val_y = torch.tensor(val_l).long()
    test_y = val_y

    return train_x, train_y, val_x, val_y, test_x, test_y, vocab_size