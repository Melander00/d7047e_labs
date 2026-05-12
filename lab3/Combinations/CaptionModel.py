from torch import nn

class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, captions):
        x = self.encoder(inputs)
        x = self.decoder(x, captions)
        return x

    def generate_caption(self, image, vocab, max_length = 20):
        features = self.encoder(image)
        caption = self.decoder.generate_caption(features, vocab, max_length)
        return caption