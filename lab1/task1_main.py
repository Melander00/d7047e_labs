import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader,TensorDataset
from Task1.task1_data import data_loading_code as loader
from training.training import run_training, run_model,run_test,develop_model

from Task1.Models.Basic_ANN import simpleANN
from Task1.Models.Lstm_model import LSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def main_simple_ann(save_board=False):#normal data from the lab, needs rewrite for bigger
    #add data
    train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data(text_with_data="amazon_cells_labelled_LARGE_25K")#used for the smaller dataset
   # Larger data: amazon_cells_labelled_LARGE_25K

    
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
    lr=1e-4
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    #num epochs
    epochs=100
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
    
    
    
    print(elapsed_time_seconds)
    
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
    train_data,train_labels,val_data,val_labels,vocab =loader.load_prep_data_lstm(text_with_data="amazon_cells_labelled_LARGE_25K")#used for the smaller dataset
    #Larger data: "amazon_cells_labelled_LARGE_25K"

    
    dataset_train=TensorDataset(train_data.long(),train_labels)
    dataset_val=TensorDataset(val_data.long(),val_labels)

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
    epochs=100
    #criterion(loss functions)
    criterion=nn.CrossEntropyLoss()
    #train
    best_model, losses, accs, last_model, best_val_loss, elapsed_time_seconds=run_training(model=model,
                                        loaders=loaders,
                                        criterion=criterion,
                                        optimizer=optim,
                                        num_epochs=epochs,
                                        model_name="steve_butLSTM"
    )
    #validate
    #save
    
    
    
    print(elapsed_time_seconds)
    print(len(losses[0]))
    print(len(accs[0]))
    
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










main_LSTM(True)
#main_simple_ann()
#tensorboard --logdir=runs_task1 (call this i the terminal to activate tensorboard)