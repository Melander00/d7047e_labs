from dataset import get_loaders
import torch
from Models.CNN import CNN
from Models.RNN import CaptionRNN
import Training





loader=get_loaders("Data")
val=loader[0][1]#NOTE change the last index here to 2 when you want to use the test set
vocab=loader[1].vocab
print(len(vocab))
vocab_size=len(vocab)

dataset=loader[1]

#NOTE: ----------- RUN THE FOLLOWING TO TRAIN A MODEL---------
Training.Trainmodel(model_name="steve2",vocab_size=vocab_size,save_best=True,epochs=25
                    ,model_cnn=CNN(),model_RNN=CaptionRNN(vocab_size=vocab_size),save_metadata=True)
#----------------------------------------

#NOTE-------------------------Here you run the trained model

'''run=Training.runsavedmodel(vocab=vocab,modelname="testing",model_CNN=CNN(),model_RNN=CaptionRNN(vocab_size=vocab_size),data_loader=val)
i=2 #NOTE choose which index to review. you can make a loop here to run through all the words: len(run[0])

pred=run[0][i]
real=run[1][i]

print("generated:", pred)
print("real: ", real )
#Training.wordprint(dataset=dataset,sample_pred=pred[0],sample_target=real[0])'''