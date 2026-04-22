
import torch
import torch.nn as nn
import torch.optim as optim
from Task4.Data.Dataloader import dataloader
from Task4.Model.CNN import CNN
import os
import time
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("you are using: ",device)
if device=="cpu":
    print("using cpu? really?")








def Trainmodel( save_metadata:bool ,save_best: bool, epochs:int, model:nn.Module=CNN(), model_name:str ="steve"):
    train_loader, val_loader,_=dataloader()
    #------metaparameters:
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=1e-3)

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
    #--------------------training loop.
        for (data,label) in train_loader:
            data=data.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            
            output=model(data)
            loss=criterion(output,label)
            
            loss.backward()
            optimizer.step()
            pred=output.argmax(dim=1)
            
            train_correct+=(pred==label).sum().item()
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
        model.eval()


        with torch.no_grad():
            for (data, label) in val_loader:
                data=data.to(device)
                label=label.to(device)
                
                output=model(data)
                loss = criterion(output, label)
                pred = output.argmax(dim=1)
                val_correct+=(pred==label).sum().item()
                val_total+=label.size(0)
                
                valloss += loss.item()

        val_acc=val_correct/val_total
        val_accs.append(val_acc)
        valloss /= len(val_loader)
        val_losses.append(valloss)
        
        if valloss<best_valloss and save_best:
            os.makedirs("Task4/Model/"+model_name, exist_ok=True)
            torch.save(model.state_dict(),"Task4/Model/"+model_name+"/"+model_name+".pth")

        
        
        
        #------print-outs

        print(epoch+1)
        print(train_loss,": ", train_acc)
        print(valloss,": ", val_acc)
    traintime=time.time()-start
    print("traintime: ", traintime)
    if save_metadata:
        pass
    



def runsavedmodel(modelname:str, data_loader:DataLoader, model:nn.Module=CNN()):
    print("testing model")
    
    path="Task4/Model/"+modelname+"/"+modelname+".pth"
    model.load_state_dict(torch.load(path))
    model=model.to(device)
    model.eval()
    predictions=[]
    real_labels=[]
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            pred = output.argmax(dim=1)
            predictions.append(pred)
            real_labels.append(label)




    return predictions, real_labels