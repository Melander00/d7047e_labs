
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loaders as loaders
import os
import time
import json
from torch.utils.data import DataLoader
from Models.CNN import CNN
from Models.RNN import CaptionRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("you are using: ",device)
if device=="cpu":
    print("using cpu? really?")

savepath="SavedModels"

def wordprint(sample_pred,sample_target,dataset):

    sample_pred 
    sample_target 

    pred_words = []
    target_words = []

    for idx in sample_pred:

        word = dataset.vocab.itos[idx.item()]

        if word == "<EOS>":
            break

        if word not in ["<PAD>", "<SOS>"]:
            pred_words.append(word)

    for idx in sample_target:

        word = dataset.vocab.itos[idx.item()]

        if word == "<EOS>":
            break

        if word not in ["<PAD>", "<SOS>"]:
            target_words.append(word)
    print("-----------------------------")
    print("PRED :", " ".join(pred_words))
    print("TRUE :", " ".join(target_words))
    print("-----------------------------")


def Trainmodel(vocab_size:int, save_metadata:bool ,save_best: bool, epochs:int, model_cnn:nn.Module,model_RNN:nn.Module, model_name:str ="steve"):
    print("checkpoint 1")
    (train_loader, val_loader,_),dataset=loaders("Data")
    print("checkpoint 2")
    #------metaparameters:
    model_cnn=model_cnn.to(device)
    model_RNN=model_RNN.to(device)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    criterion=nn.CrossEntropyLoss(ignore_index=pad_idx)
    lr=1e-3
    #optimizer=optim.Adam(model_RNN.parameters(), lr=lr)

    optimizer = optim.Adam(
    list(model_cnn.parameters()) +
    list(model_RNN.parameters()),
    lr=lr
    )





    best_valloss=float('inf')
    train_accs=[]
    val_accs=[]
    val_losses=[]
    train_losses=[]
    start=time.time()
    for epoch in range(epochs):
        train_loss=0
        train_correct=0
        train_total=0
        model_cnn.train()
        model_RNN.train()
    #--------------------training loop.
        for (data,label) in train_loader:
            data=data.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            
            
            #We use the feature array from the CNN as input for the RNN


            # teacher forcing
            inputs = label[:, :-1]
            targets = label[:, 1:]

            features=model_cnn(data)
            output=model_RNN(features,inputs)
            output = output[:, 1:, :]
            
            loss=criterion(output.reshape(-1,vocab_size),targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            pred=output.argmax(dim=2)
            
            train_correct+=(pred==targets).sum().item()
            train_total+=label.size(0)
            train_loss += loss.item()
        
        train_loss/=len(train_loader)
        train_acc=train_correct/train_total
        train_accs.append(train_acc)
        train_losses.append(train_loss)

    #------------------validationloop
        valloss=0
        val_correct=0
        val_total=0
        model_cnn.eval()
        model_RNN.eval()


        with torch.no_grad():
            for (data, label) in val_loader:
                data=data.to(device)
                label=label.to(device)
                inputs = label[:, :-1]
                targets = label[:, 1:]
                features=model_cnn(data)
                output=model_RNN(features,inputs)
                output = output[:, 1:, :]
                
                
                loss=criterion(output.reshape(-1,vocab_size),targets.reshape(-1))
                pred = output.argmax(dim=2)
                val_correct+=(pred==targets).sum().item()
                val_total+=label.size(0)
                wordprint(sample_pred=pred[0],sample_target=targets[0],dataset=dataset)
                valloss += loss.item()

        val_acc=val_correct/val_total
        val_accs.append(val_acc)
        valloss /= len(val_loader)
        val_losses.append(valloss)
        
        if valloss<best_valloss and save_best:
            best_valloss=valloss
            best_epoch=epoch+1
            os.makedirs(os.path.join(savepath,model_name), exist_ok=True)
            torch.save(model_RNN.state_dict(),os.path.join(savepath,model_name,model_name+"_RNN.pth"))
            torch.save(model_cnn.state_dict(),os.path.join(savepath,model_name,model_name+"_CNN.pth"))

        
        
        
        #------print-outs

        print(epoch+1)
        print(train_loss,": ", train_acc)
        print(valloss,": ", val_acc)
    traintime=time.time()-start
    print("traintime: ", traintime)
   
    if save_metadata:
        data_meta={"traintime":traintime,
                    "best_epoch": best_epoch,
                    "best_valloss":best_valloss,
                    "epochs":epochs,
                    "lr":lr,
                    "val_losses":val_losses,
                    "train_losses":train_losses,
                    "val_acc":val_accs,
                    "train_accs":train_accs}
        path_meta=os.path.join(savepath,model_name,"Metadata.json")
        with open(path_meta, "w") as f:
            json.dump(data_meta, f)
            print("Meta-data saved")
    





#TODO add a parameter for the cnn and rnn model used, and load them respectivally 
def runsavedmodel(modelname:str, data_loader:DataLoader, model_CNN:nn.Module,model_RNN:nn.Module):
    print("testing model")
    
    path_CNN=os.path.join(savepath,modelname,modelname+"_CNN.pth")
    path_RNN=os.path.join(savepath,modelname,modelname+"_RNN.pth")
    model_CNN.load_state_dict(torch.load(path_CNN))
    model_RNN.load_state_dict(torch.load(path_RNN))
   
    model_CNN=model_CNN.to(device)
    model_RNN=model_RNN.to(device)
    model_CNN.eval()
    model_RNN.eval()


    predictions=[]
    real_labels=[]
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            inputs = label[:, :-1]

            features = model_CNN(data)

            output = model_RNN(features, inputs)

            output = output[:, 1:, :]

            pred = output.argmax(dim=2)

            predictions.append(pred)

            real_labels.append(label[:, 1:])




    return predictions, real_labels