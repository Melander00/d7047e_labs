from torch import nn

from Models.Attention import ResNetCNN, DecoderRNN
from Combinations.CaptionModel import CaptionModel


class AttentionCaption(CaptionModel):
    def __init__(self, vocab_size):
        super().__init__(ResNetCNN(), DecoderRNN(
            vocab_size=vocab_size,
        ))