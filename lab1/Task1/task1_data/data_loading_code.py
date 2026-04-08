import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import os

print(os.getcwd())

def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_.loc[len(df_)] = {
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent)
        }
    return data

# If this is the primary file that is executed (ie not an import of another file)
#if __name__ == "__main__":
def load_prep_data(text_with_data="amazon_cells_labelled.txt"):
    
    if text_with_data=="amazon_cells_labelled.txt":
        print("small data generated")
    else:
        print("larger data simpleANN")



    # get data, pre-process and split
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, text_with_data)


    data = pd.read_csv(file_path, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                             # pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )

    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=20000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()
    return train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor,word_vectorizer.vocabulary_


def load_prep_data_lstm(text_with_data="amazon_cells_labelled.txt"):
    if text_with_data=="amazon_cells_labelled.txt":
        print("small data generated (LSTM)")
    else:
        print("larger data LSTM")
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, text_with_data)

    data = pd.read_csv(file_path, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    training_data, validation_data, training_labels, validation_labels = train_test_split(
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )

    # ---------- NEW PART (replaces TF-IDF) ----------

    def tokenize(text):
        return text.lower().split()

    # build vocab
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in training_data:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vocab_size = len(vocab)

    # encode + pad
    max_len = 20

    def encode(text):
        return [vocab.get(w, vocab["<UNK>"]) for w in tokenize(text)]

    def pad(seq):
        return seq[:max_len] + [0] * (max_len - len(seq))

    training_data = [pad(encode(t)) for t in training_data]
    validation_data = [pad(encode(t)) for t in validation_data]

    # ---------- SAME AS BEFORE ----------
    train_x_tensor = torch.tensor(training_data).long()
    train_y_tensor = torch.tensor(training_labels).long()

    validation_x_tensor = torch.tensor(validation_data).long()
    validation_y_tensor = torch.tensor(validation_labels).long()

    return train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size