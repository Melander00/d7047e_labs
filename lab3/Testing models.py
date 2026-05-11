from dataset import get_loaders
import torch
from Models.CNN import CNN
from Models.RNN import CaptionRNN
import Training

'''A=get_loaders("Data")
Dataloader=A[0][0]
subset=Dataloader.dataset
print(subset[0][0].shape)'''
vocab=get_loaders("Data")[1].vocab
print(len(vocab))
vocab=len(vocab)

Training.Trainmodel(vocab_size=vocab,save_best=True,epochs=1,model_cnn=CNN(),model_RNN=CaptionRNN(vocab_size=vocab),save_metadata=True)