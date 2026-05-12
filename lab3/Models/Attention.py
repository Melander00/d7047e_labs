from typing import Any

import torch
from torch import nn
from torchvision import models

class ResNetCNN(nn.Module):

    def __init__(self, freeze_backbone=True):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        for param in resnet.parameters():
            param.requires_grad = not freeze_backbone

        # remove avgpool + fc
        modules = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*modules)

    def forward(self, images):

        features = self.resnet(images)

        # features:
        # (B, 2048, 7, 7)

        batch_size = features.size(0)

        # flatten spatial dimensions
        features = features.permute(0, 2, 3, 1)

        # (B, 7, 7, 2048)

        features = features.view(
            batch_size,
            -1,
            2048
        )

        # (B, 49, 2048)

        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super().__init__()

        self.att_enc = nn.Linear(encoder_dim, attention_dim)
        self.att_dec = nn.Linear(hidden_dim, attention_dim)

        self.att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, hidden_state):
        att1 = self.att_enc(encoder_out)
        att2 = self.att_dec(hidden_state).unsqueeze(1)

        att = self.att(
            self.relu( att1 + att2 )
        ).squeeze(2)

        alpha = self.softmax(att)

        attention_encoding = (
            encoder_out * alpha.unsqueeze(2)
        ).sum(dim=1)

        return attention_encoding, alpha


class DecoderRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        attention_dim=512,
        encoder_dim=2048,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.attention = Attention(
            encoder_dim,
            hidden_dim,
            attention_dim
        )

        self.lstm = nn.LSTMCell(
            embed_dim + encoder_dim,
            hidden_dim
        )

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out, captions):

        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)

        embeddings = self.embedding(captions)

        mean_encoder = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)

        outputs = torch.zeros(
            batch_size,
            seq_len + 1,
            self.fc.out_features,
            device=captions.device
        )

        # timestep 0 = image-only step
        attention_context, alpha = self.attention(
            encoder_out,
            h
        )

        zero_embed = torch.zeros(
            batch_size,
            embeddings.size(2),
            device=captions.device
        )

        lstm_input = torch.cat(
            [zero_embed, attention_context],
            dim=1
        )

        h, c = self.lstm(lstm_input, (h, c))

        outputs[:, 0, :] = self.fc(self.dropout(h))

        # caption steps
        for t in range(seq_len):

            attention_context, alpha = self.attention(
                encoder_out,
                h
            )

            emb_t = embeddings[:, t, :]

            lstm_input = torch.cat(
                [emb_t, attention_context],
                dim=1
            )

            h, c = self.lstm(
                lstm_input,
                (h, c)
            )

            preds = self.fc(self.dropout(h))

            outputs[:, t + 1, :] = preds

        return outputs
    
    def generate_caption(self, encoder_out, vocab, max_length=20):

        result = []

        batch_size = encoder_out.size(0)

        device = encoder_out.device

        # initialize hidden states
        mean_encoder = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)

        # first input = zero embedding
        inputs = torch.zeros(
            batch_size,
            self.embedding.embedding_dim,
            device=device
        )

        for _ in range(max_length):

            # attention over image features
            attention_context, alpha = self.attention(
                encoder_out,
                h
            )

            # concatenate embedding + attention context
            lstm_input = torch.cat(
                [inputs, attention_context],
                dim=1
            )

            # LSTM step
            h, c = self.lstm(
                lstm_input,
                (h, c)
            )

            # vocabulary prediction
            output = self.fc(self.dropout(h))

            # best token
            predicted = output.argmax(dim=1)

            predicted_idx = predicted.item()

            word = vocab.itos[predicted_idx]

            if word == "<EOS>":
                break

            if word not in ["<SOS>", "<PAD>"]:
                result.append(word)

            # embedding of predicted token becomes next input
            inputs = self.embedding(predicted)

        return " ".join(result)