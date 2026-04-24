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
import matplotlib.pyplot as plt
import copy
import random
#.\.venv\Scripts\activate
modelname="steve2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("Task4/Model/"+modelname+"/"+modelname+".pth"):
    print("model not found...")
    answer = input("do you wish to train a new one? (y/n)")
    if answer.lower()=="y":
        Trainmodel(epochs=10,save_best=True,model_name=modelname,save_metadata=False)
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
    #index_used=indices_list[0]
    #print(index_used)
    #img,_=testdata.dataset[index_used]
    #img:Tensor
    #visualize_image(img)
       
       
       
    #perfrom the attack here
    model
    model.load_state_dict(torch.load("Task4/Model/"+modelname+"/"+modelname+".pth"))  
    model.eval()
    predictions=[]
    predictions_att=[]
    pics=[]
    attacked_pics=[]
    confidence=[]
    confidence_attack=[]
    for i in range(3):

        index_used=random.choice(indices_list)
        img,_=testdata.dataset[index_used]
        img = img.unsqueeze(0)
        img.requires_grad_(True)
        F=nn.functional
        output=model(img)
        pred = output.argmax(dim=1)
        probs = F.softmax(output, dim=1)
        conf, _ = probs.max(dim=1)

        
        print("before attack the picture is predicted as: ", pred.item())
        


        target = torch.tensor([9])


        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        
        epsilon = 0.25
        perturbed = img - epsilon * img.grad.sign()
        perturbed = torch.clamp(perturbed, 0, 1)

        output_attack=model(perturbed)
        pred_attack = output_attack.argmax(dim=1)
        probs_att = F.softmax(output_attack, dim=1)
        conf_att, _ = probs_att.max(dim=1)
        
        
        predictions.append(pred.item())
        predictions_att.append(pred_attack.item())
        
        pics.append(img)
        attacked_pics.append(perturbed)
        confidence.append(round(conf.item(),3))
        confidence_attack.append(round(conf_att.item(),3))



        print("After attack the picture is predicted as: ", pred_attack.item())

    visualize=input("do you wish to visualize the before and after pictures? (y/n)")
    if visualize.lower()=="y":
        plt.Figure()
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.suptitle("Adveserial attack on images")
        for i in range(len(attacked_pics)):

            before="Before attack\n"+"prediction: " +str(predictions[i])+"\n confidance: "+ str(confidence[i])
            after="After attack\n"+"prediction: "+str(predictions_att[i])+"\n confidance: "+ str(confidence_attack[i])
        
            plt.subplot(3, 2, 2*i+1)
            visualize_image(pics[i].squeeze().detach(),title=before)
            plt.subplot(3, 2, 2*i+2)

            visualize_image(attacked_pics[i].squeeze().detach(), title=after)
        plt.show()




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
    
    
    
    optimizer=optim.Adam([noise],lr=0.01)
    Pictures=[]
    predictions=[]
    confidance=[]
    for i in range(8):
        optimizer.zero_grad()
        output=model(noise)

        Pictures.append(copy.deepcopy(noise))

        prediction=output.argmax(dim=1).item()
        predictions.append(prediction)
        F=nn.functional
        probs = F.softmax(output, dim=1)
        conf, _ = probs.max(dim=1)
        confidance.append(round(conf.item(),3))

        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        

        if i%1==0:
            #print("prediction: ", prediction)
            #visualize_image(noise.squeeze().detach())
            pass
    
    print("final prediction: ",prediction)
    inp=input("do you wish to visualize the random noise attack?(y/n)")
    
    if inp.lower()=="y":
        plt.figure()
        plt.suptitle("Random noise attack")
        plt.subplots_adjust(hspace=1, wspace=0.2)
        for i in range(len(Pictures)):
            noised =Pictures[i]
            plt.subplot(4, 2, i+1)
            ti=f"epoch: {i+1}\n"+"Prediction: "+str(predictions[i])+"\n Confidence: "+str(confidance[i])
            visualize_image(noised.squeeze().detach(),title=ti) 
        plt.show()



if __name__=="__main__":
    print("first step, making a advesary attack")
    #Step1()
    print("second step. changing random noise until the model predicts it")
    step2()