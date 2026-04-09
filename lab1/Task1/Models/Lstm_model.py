import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        print("LSTM constructed")

        # Original 5-layer LSTM that worked well
        self.embedding = nn.Embedding(vocab_size, 30)
        self.lstm = nn.LSTM(30, 64, batch_first=True, num_layers=5, dropout=0.3)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]  # Take the last layer's hidden state
        out = self.dropout(out)
        out = self.fc1(out)
        return out