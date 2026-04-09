import os

import numpy as np
import pandas as pd
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load stopwords once at module level — fixes the performance bug of loading inside the loop
STOPWORDS = set(stopwords.words('english'))


def preprocess_pandas(data, columns, remove_stopwords=True):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)           # remove emails
    data['Sentence'] = data['Sentence'].replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)  # remove IPs
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)                                # remove special chars
    data['Sentence'] = data['Sentence'].replace(r'\d', '', regex=True)                                        # 2-layer LSTM: 64 input → 128 hidden (reduced from 5 layers which was excessive)
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        if remove_stopwords:
            filtered_sent = [w for w in word_tokens if w not in STOPWORDS]
        else:
            filtered_sent = word_tokens
        df_.loc[len(df_)] = {
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent)
        }
    return df_


def load_prep_data(text_with_data="amazon_cells_labelled.txt"):
    """
    Loads and preprocesses the Amazon dataset for the Simple ANN (TF-IDF).
    Returns a 3-way split: train / val / test (80% / 10% / 10%)
    Returns: train_x, train_y, val_x, val_y, test_x, test_y, vocabulary
    """
    if text_with_data == "amazon_cells_labelled.txt":
        print("small data generated")
    else:
        print("larger data simpleANN")

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, text_with_data)

    data = pd.read_csv(file_path, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    sentences = data['Sentence'].values.astype('U')
    labels    = data['Class'].values.astype('int32')

    # 3-way split: 80% train, 10% val, 10% test
    train_s, temp_s, train_l, temp_l = train_test_split(
        sentences, labels, test_size=0.20, random_state=0, shuffle=True
    )
    val_s, test_s, val_l, test_l = train_test_split(
        temp_s, temp_l, test_size=0.50, random_state=0, shuffle=True
    )

    # TF-IDF: fit ONLY on training data to avoid data leakage
    word_vectorizer = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        max_features=20000, max_df=0.5,
        use_idf=True, norm='l2'
    )
    train_x = torch.from_numpy(np.array(word_vectorizer.fit_transform(train_s).todense())).float()
    val_x   = torch.from_numpy(np.array(word_vectorizer.transform(val_s).todense())).float()
    test_x  = torch.from_numpy(np.array(word_vectorizer.transform(test_s).todense())).float()

    train_y = torch.from_numpy(np.array(train_l)).long()
    val_y   = torch.from_numpy(np.array(val_l)).long()
    test_y  = torch.from_numpy(np.array(test_l)).long()

    return train_x, train_y, val_x, val_y, test_x, test_y, word_vectorizer.vocabulary_


def load_prep_data_lstm(text_with_data="amazon_cells_labelled.txt"):
    """
    Loads and preprocesses the Amazon dataset for the LSTM (custom tokenizer).
    Returns a 3-way split: train / val / test (80% / 10% / 10%)
    Returns: train_x, train_y, val_x, val_y, test_x, test_y, vocab_size
    """
    if text_with_data == "amazon_cells_labelled.txt":
        print("small data generated (LSTM)")
    else:
        print("larger data LSTM")

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, text_with_data)

    data = pd.read_csv(file_path, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns, remove_stopwords=False)

    sentences = data['Sentence'].values.astype('U')
    labels    = data['Class'].values.astype('int32')

    # 3-way split: 80% train, 10% val, 10% test
    train_s, temp_s, train_l, temp_l = train_test_split(
        sentences, labels, test_size=0.20, random_state=0, shuffle=True
    )
    val_s, test_s, val_l, test_l = train_test_split(
        temp_s, temp_l, test_size=0.50, random_state=0, shuffle=True
    )

    def tokenize(text):
        return text.lower().split()

    # Build vocabulary ONLY from training data to avoid data leakage
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in train_s:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vocab_size = len(vocab)
    max_len = 64  # Increased from 20 — captures more context per review

    def encode_and_pad(text):
        ids = [vocab.get(w, vocab["<UNK>"]) for w in tokenize(text)]
        return ids[:max_len] + [0] * max(0, max_len - len(ids))

    train_x = torch.tensor([encode_and_pad(t) for t in train_s]).long()
    val_x   = torch.tensor([encode_and_pad(t) for t in val_s]).long()
    test_x  = torch.tensor([encode_and_pad(t) for t in test_s]).long()

    train_y = torch.tensor(train_l).long()
    val_y   = torch.tensor(val_l).long()
    test_y  = torch.tensor(test_l).long()

    return train_x, train_y, val_x, val_y, test_x, test_y, vocab_size