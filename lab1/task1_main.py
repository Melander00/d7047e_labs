import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset.loader import prepare_yelp_loaders
from torch.utils.data import DataLoader,TensorDataset
from Task1.task1_data import data_loading_code as loader
from training.training import run_training, run_model,run_test,develop_model
import time
from Task1.Models.Basic_ANN import simpleANN
from Task1.Models.Lstm_model import LSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createcache(CACHE_PATH = "ANN_cached_data.pt",text_with_data="amazon_cells_labelled_LARGE_25K.txt"):
        print("Processing raw dataset...")
            
        train_data,train_labels,val_data,val_labels,vocab = loader.load_prep_data(
                text_with_data="amazon_cells_labelled_LARGE_25K.txt"
            )

        torch.save((train_data, train_labels, val_data, val_labels, vocab), CACHE_PATH)
        print("Saved dataset to cache.")










def main_simple_ann(save_board=False):#normal data from the lab, needs rewrite for bigger
    #add data
    #train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data(text_with_data="amazon_cells_labelled_LARGE_25K.txt")#used for the smaller dataset
   # Larger data: amazon_cells_labelled_LARGE_25K.txt

    CACHE_PATH = "ANN_cached_data.pt"
    text_with_data="amazon_cells_labelled_LARGE_25K.txt"
    #text_with_data="amazon_cells_labelled.txt"
    if text_with_data=="amazon_cells_labelled_LARGE_25K.txt":#creates a cache for the 25k dataset to sped up
        if os.path.exists(CACHE_PATH):
            print("Loading cached dataset...")
            train_data, train_labels, val_data, val_labels, vocab = torch.load(CACHE_PATH,weights_only=False)
        

    else :
        train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data()











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
    
    #add optimizer
    lr=1e-3
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    #num epochs
    epochs=10
    #criterion(loss functions)
    criterion=nn.CrossEntropyLoss()
    #train
    best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds=run_training(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="steve"
    )
    #validate
    #save
    
    
    
    
    
    if save_board:
        writer = SummaryWriter("runs_task1/ANN")

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




def main_LSTM(save_board=False):#normal data from the lab, needs rewrite for bigger
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
    epochs=10
    #criterion(loss functions)
    criterion=nn.CrossEntropyLoss()
    #train
    start = time.time()
    best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds=run_training(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="steve_butLSTM"
    )
    #validate
    #save
    end = time.time()

    print("Total training time:", end - start)
    
    
    
    
    
    
    if save_board:
        writer = SummaryWriter("runs_task1/LSTM")

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


def main_bigdata(Model="LSTM"):
    


    (traindata,valdata,testdata), dataset = prepare_yelp_loaders(batch_size=64)

    #makes a vocab based on trainingdata:
    max_id = 0
    for data, _ in traindata:
        max_id = max(max_id, data.max().item())

    vocab_size = max_id + 1
    print("Vocab size:", vocab_size)

    if Model=="LSTM":
        model=LSTM(vocab_size).to(device)
    elif Model=="ANN":
        model=simpleANN(vocab_size).to(device)
    
    loaders=(traindata,valdata,testdata)
    criterion=nn.CrossEntropyLoss()
    lr=1e-4
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    epochs=1





    best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds=run_training(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="steve but big"
    )
   
   
    print(elapsed_time_seconds)  






#main_LSTM()
main_simple_ann()
#main_bigdata()
#createcache()
#tensorboard --logdir=runs_task1 (call this i the terminal to activate tensorboard)