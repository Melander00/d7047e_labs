from Task4.Data.Dataloader import dataloader
from Task4.trainmodel import Trainmodel, runsavedmodel
from Task4.Model.CNN import CNN
from Task4.Visualize import visualize_image
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import json
#.\.venv\Scripts\activate
modelname="steve"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("Task4/Model/"+modelname+"/"+modelname+".pth"):
    print("model not found...")
    answer = input("do you wish to train a new one? (y/n)")
    if answer.lower()=="y":
        Trainmodel(epochs=5,save_best=True,model_name=modelname,save_metadata=False)
    else:
        print("No model trained")
        print("A model has to be defined and trained in order to proceed...")
        exit()




#step2-------------
    #we have the model, now we will try to fool it to predict 9, using an adveserial attack


def Step1(model_name=modelname,model:nn.Module=CNN()):
    path="Task4/Model/"+modelname+"/"+"fourindex.json"
    _,_,testdata=dataloader()
 #--------check for  pred==real and they are 4, return index and use that to find the picture we want to attack
    if not os.path.exists(path):
        print("finding correct 4:s")
        predlist,reallist= runsavedmodel(modelname=model_name,data_loader=testdata)
        labels_all = torch.cat(reallist)
        preds_all  = torch.cat(predlist)
        
        indices = torch.where((labels_all == 4) & (preds_all == 4))[0]
        indices_list = indices.tolist()        
        data = {"indices": indices_list}

        with open(path, "w") as f:
            json.dump(data, f)
            
    with open(path, "r") as f:
        data = json.load(f)

    indices_list = data["indices"]

    #print("With " + modelname+": " + " predicted 4 correctly ", len(indices_list), " times")
#--------------chooses a index to load our picture, this is the one we assault
    index_used=indices_list[0]
    #print(index_used)
    img,_=testdata.dataset[index_used]
    img:Tensor
    #visualize_image(img)
       
       
       
    #perfrom the attack here
    model
    model.load_state_dict(torch.load("Task4/Model/"+modelname+"/"+modelname+".pth"))  
    model.eval()
    img = img.unsqueeze(0)
    img.requires_grad_(True)
    
    output=model(img)
    pred = output.argmax(dim=1)
    print("before attack the picture is predicted as: ", pred.item())


    target = torch.tensor([9])


    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    model.zero_grad()
    loss.backward()
    
    epsilon = 0.2
    perturbed = img - epsilon * img.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    output_attack=model(perturbed)
    pred_attack = output_attack.argmax(dim=1)
    print("After attack the picture is predicted as: ", pred_attack.item())

    visualize=input("do you wish to visualize the picture before and after? (y/n)")
    if visualize.lower()=="y":
        print("first pic is the before. Close it to see the after!")
        visualize_image(img.squeeze().detach())
        visualize_image(perturbed.squeeze().detach())




#Step1()






def step2(modelname:str=modelname,model:nn.Module=CNN()):
    #create a attack using random noise should be run after step1
    path_to_model="Task4/Model/"+modelname+"/"+modelname+".pth"
    noise=torch.rand((1,1,28,28),requires_grad=True)# random pciture
    target=torch.Tensor([9]).long()#the target
    criterion=nn.CrossEntropyLoss()
    model
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    
    
    
    optimizer=optim.Adam([noise],lr=0.1)
    for i in range(100):
        optimizer.zero_grad()
        output=model(noise)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print("prediction: ", output.argmax(dim=1).item())
    print("final prediction: ",output.argmax(dim=1).item( ))
    inp=input("do you wish to visualize the random noise attack?(y/n)")
    if inp.lower()=="y":
        visualize_image(noise.squeeze().detach())



if __name__=="__main__":
    print("first step, making a advesary attack")
    Step1()
    print("second step. changing random noise until the model predicts it")
    step2()