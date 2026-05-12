from Models.CNN import CNN
from Models.RNN import CaptionRNN
from Combinations.CaptionModel import CaptionModel

class BaseCaption(CaptionModel):
    def __init__(self, vocab_size):
        super().__init__(CNN(), CaptionRNN(vocab_size=vocab_size))
