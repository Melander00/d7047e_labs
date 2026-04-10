import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset.loader import prepare_yelp_loaders
from torch.utils.data import DataLoader,TensorDataset
from Task1.task1_data import data_loading_code as loader
from training.training import run_training, run_model,run_test,develop_model, save_model
import time
from Task1.Models.Basic_ANN import simpleANN
from Task1.Models.Lstm_model import LSTM
from transformers import AutoModel, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#----------cache function. do not use unless the time for loading data is unreasonble
def createcache(CACHE_PATH = "ANN_cached_data.pt",text_with_data="amazon_cells_labelled_LARGE_25K.txt"):
        print("Processing raw dataset...")
            
        train_data,train_labels,val_data,val_labels,vocab = loader.load_prep_data_lstm(
                text_with_data
            )#change this for ANN

        torch.save((train_data, train_labels, val_data, val_labels, vocab), CACHE_PATH)
        print("Saved dataset to cache.")


#saves data into a tensorboard, can be used in the other functions
def saveboard(losses:list, accs: list, Model:str, ):
    print("saving tensorboard")
    writer = SummaryWriter(f"runs_task1/{Model}")

    train_losses = losses[0]
    val_losses = losses[1]

    train_accs = accs[0]
    val_accs = accs[1]

    for epoch in range(len(train_losses)):
        writer.add_scalar("Loss/Train", train_losses[epoch], epoch)
        writer.add_scalar("Loss/Val", val_losses[epoch], epoch)

        writer.add_scalar("Accuracy/Train", train_accs[epoch], epoch)
        writer.add_scalar("Accuracy/Val", val_accs[epoch], epoch)
    writer.close()
    
    
    
    



#-------------------amazon dataset--------------
#------------------ANN----------

def main_simple_ann(save_board=False):
    #add data
    #train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data(text_with_data="amazon_cells_labelled_LARGE_25K.txy")#used for the smaller dataset
   # Larger data: amazon_cells_labelled_LARGE_25K.txy

    CACHE_PATH = "ANN_cached_data.pt"
    text_with_data="amazon_cells_labelled_LARGE_25K.txt"
    #text_with_data="amazon_cells_labelled.txt"
    if text_with_data=="amazon_cells_labelled_LARGE_25K.txt":#creates a cache for the 25k dataset to sped up
        if os.path.exists(CACHE_PATH):
            print("Loading cached dataset...")
            train_data, train_labels, val_data, val_labels, vocab = torch.load(CACHE_PATH,weights_only=False)
        

    else :
        print("amazon_cells_labelled.txt")
        train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data()





#----------makes datalaoder from the tensors
    dataset_train=TensorDataset(train_data,train_labels)
    dataset_val=TensorDataset(val_data,val_labels)

    dataloader_training=DataLoader(batch_size=64,shuffle=True,dataset=dataset_train)
    dataloader_validation=DataLoader(batch_size=64,shuffle=True,dataset=dataset_val)
    dataloader_testing=DataLoader([])
    loaders=(dataloader_training,dataloader_validation,dataloader_testing)
    #vocab and vocabsize
    vocab_size=len(vocab)
    #choose model
    model=simpleANN(vocab_size).to(device)
    #--------------meta parameters
    #add optimizer
    lr=1e-3
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    #num epochs
    epochs=80
    #criterion(loss functions)
    criterion=nn.CrossEntropyLoss()
    #-------------------------------
    #train
    metadata=develop_model(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="steve_big_amazon",
                                        
    )
    #validate
    #save
    print(metadata)
    losses= (metadata['train_loss'],metadata['val_loss'])
    accs=(metadata['train_accuracy'],metadata['val_accuracy'])
    if save_board:
        saveboard(accs=accs,losses=losses,Model="Ann_25k")



    
    
    
    
#-----------------for the LSTM--------------


def main_LSTM(save_board=False,Save_model=False):
    #add data
    #train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data_lstm(text_with_data="amazon_cells_labelled_LARGE_25K.txt")#used for the smaller dataset
    #Larger data: "amazon_cells_labelled_LARGE_25K.txt"
    
    
    CACHE_PATH = "lstm_cached_data.pt"
    text_with_data="amazon_cells_labelled_LARGE_25K.txt"
    #text_with_data="amazon_cells_labelled.txt"
    if text_with_data=="amazon_cells_labelled_LARGE_25K.txt":
        if os.path.exists(CACHE_PATH):
            print("Loading cached dataset...")
            train_data, train_labels, val_data, val_labels, vocab = torch.load(CACHE_PATH)
        

    else :
        train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data_lstm()


    print("vocab_size=",vocab)





    
    dataset_train=TensorDataset(train_data,train_labels)
    dataset_val=TensorDataset(val_data,val_labels)

    dataloader_training=DataLoader(batch_size=64,shuffle=True,dataset=dataset_train)
    dataloader_validation=DataLoader(batch_size=64,shuffle=True,dataset=dataset_val)
    dataloader_testing=DataLoader([])
    loaders=(dataloader_training,dataloader_validation,dataloader_testing)
    #vocab and vocabsize
    #vocab_size=len(vocab)
    #choose model
    model=LSTM(vocab).to(device)
    
    #add optimizer
    lr=1e-3
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    #num epochs
    epochs=80
    #criterion(loss functions)
    criterion=nn.CrossEntropyLoss()
    #train
    metadata=develop_model(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="LSTM_big_amazon",
                                        
    )
    #validate
    #save
    print(metadata)
    losses= (metadata['train_loss'],metadata['val_loss'])
    accs=(metadata['train_accuracy'],metadata['val_accuracy'])
    if save_board:
        saveboard(accs=accs,losses=losses,Model="LSTM_25k")
    
    
    
    

#------for use of the bigger yelp dataset--------
def main_bigdata(Model:str="LSTM",save_board:bool=False):
    

    print("error1 check")
    #token=AutoTokenizer.from_pretrained("bert-base-uncased")
    
    
    
    Jason="reviews_500000.jsonl"
    if Model=="LSTM":
        start = time.time()
        CACHE_PATH = "LSTM500k_cached_data.pt"
        #CACHE_PATH = "LSTMbig_cached_data.pt"
       # print("Loading cached dataset...")
        
        train_data, train_labels, val_data, val_labels, vocab = torch.load(CACHE_PATH)
        #train_data, train_labels, val_data, val_labels, vocab=loader.load_prep_bigdata_lstm(Jason)
        vocab_size=vocab

        print("datload:", time.time() - start)
        model=LSTM(vocab_size).to(device)
        

    

    elif Model=="ANN":
        #CACHE_PATH
        start = time.time()
        
        
        train_data, train_labels, val_data, val_labels, vocab=loader.load_prep_bigdata_ANN(Jason)
        #train_data, train_labels, val_data, val_labels, vocab = torch.load(CACHE_PATH)
        print("datload:", time.time() - start)
        vocab_size=vocab
        model=simpleANN(vocab_size).to(device)

    dataset_train=TensorDataset(train_data,train_labels)
    dataset_val=TensorDataset(val_data,val_labels)
    
    dataloader_training=DataLoader(batch_size=64,shuffle=True,dataset=dataset_train)
    dataloader_validation=DataLoader(batch_size=64,shuffle=False,dataset=dataset_val)
    dataloader_testing=DataLoader([])

    loaders=(dataloader_training,dataloader_validation,dataloader_testing)
    
    
    criterion=nn.CrossEntropyLoss()
    lr=1e-3
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    epochs=80





    metadata=develop_model(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="LSTM_Yelp",
                                        
    )
    #validate
    #save
    print(metadata)
    losses= (metadata['train_loss'],metadata['val_loss'])
    accs=(metadata['train_accuracy'],metadata['val_accuracy'])
    if save_board:
        saveboard(accs=accs,losses=losses,Model="ANN_500k") 






#main_LSTM()
#main_simple_ann()
#start = time.time()
if __name__ == "__main__":
   # main_LSTM()
   # main_simple_ann(save_board=True)
    #main_bigdata("LSTM",save_board=True)
    pass

  #main_LSTM()
#print("total time:", time.time() - start)
#createcache(CACHE_PATH="lstm_cached_data.pt")
#python -m tensorboard.main --logdir=runs_task1 (call this i the terminal to activate tensorboard)


