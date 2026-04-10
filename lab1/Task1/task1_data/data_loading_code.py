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
import json
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
        test_size=0.15,
        random_state=0,
        train_size=0.75,
        shuffle=False
    )

    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=2000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    
    
    
    
    
    
    '''train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()'''

    train_x_tensor = torch.from_numpy(np.array(training_data, dtype=np.float32))
    train_y_tensor = torch.from_numpy(np.array(training_labels, dtype=np.int64))

    validation_x_tensor = torch.from_numpy(np.array(validation_data, dtype=np.float32))
    validation_y_tensor = torch.from_numpy(np.array(validation_labels, dtype=np.int64))






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
    
    '''sentences = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split('\t')
            sentences.append(text)
            labels.append(int(label))'''








    training_data, validation_data, training_labels, validation_labels = train_test_split(
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
       
        
        
        test_size=0.15,
        train_size=0.75,
        random_state=0,
        shuffle=False,
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




def load_prep_bigdata_lstm(jsonL="reviews_all.jsonl"):
    #load the json and format it into tensors like previous. Then, if needed, cache them with the same method as before
    
    
    print("largest dataset ",jsonL)
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, jsonL)

   
#----------------------------------------------------------------------
    
    

    sentences = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            for i,line in enumerate(f):
                item = json.loads(line)
                sentences.append(item["text"])   # or "Sentence"
                labels.append(item["label"])     # or "Class"
                if i % 100000 == 0:
                    print(f"Processed {i} samples")
    
    
    
    
   


    training_data, validation_data, training_labels, validation_labels = train_test_split(
    sentences,
    labels,
    test_size=0.15,
    train_size=0.75,
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
    
    
    
    
    
    
    


def load_prep_bigdata_ANN(jsonL="reviews_all.jsonl"):
    #load the json and format it into tensors like previous. Then, if needed, cache them with the same method as before
    
    
    print("largest dataset ",jsonL)
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, jsonL)

   
#----------------------------------------------------------------------
    
    

    sentences = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            for i,line in enumerate(f):
                item = json.loads(line)
                sentences.append(item["text"])   # or "Sentence"
                labels.append(item["label"])     # or "Class"
                if i % 100000 == 0:
                    print(f"Processed {i} samples")
    
    
    
    


    training_data, validation_data, training_labels, validation_labels = train_test_split(
    sentences,
    labels,
    test_size=0.15,
    train_size=0.75,
    )





    # ----------  TF-IDF) ----------
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx=2
    max_vocab = 1000
    def tokenize(text):
        return text.lower().split()

    # build vocab
    

    for text in training_data:
        for word in tokenize(text):
            if word not in vocab:
                if len(vocab) < max_vocab:
                    vocab[word] = idx
                    idx += 1

    vocab_size = len(vocab)

    # encode + pad
    max_len = 20
    def vectorize(text):
        vec = [0] * vocab_size
        for w in tokenize(text):
            if w in vocab:
                vec[vocab[w]] += 1
        return vec

    training_data = [vectorize(t) for t in training_data]
    validation_data = [vectorize(t) for t in validation_data]
    

    # ---------- SAME AS BEFORE ----------
    train_x_tensor = torch.tensor(training_data, dtype=torch.float16)
    train_y_tensor = torch.tensor(training_labels).long()

    validation_x_tensor = torch.tensor(validation_data, dtype=torch.float16)
    validation_y_tensor = torch.tensor(validation_labels).long()

    return train_x_tensor, train_y_tensor, validation_x_tensor, validation_y_tensor, vocab_size