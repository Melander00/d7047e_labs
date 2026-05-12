from Models.RNN import CaptionRNN
from Models.ResNet import ResNetCNN
from Combinations.CaptionModel import CaptionModel

class ResNetCaption(CaptionModel):
    def __init__(self, vocab_size):
        super().__init__(ResNetCNN(), CaptionRNN(vocab_size=vocab_size, input_features=2048))
