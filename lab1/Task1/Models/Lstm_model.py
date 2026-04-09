import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        print("LSTM constructed")

        # Embedding: maps each word ID to a 64-dim vector (up from 30)
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)

        # 2-layer LSTM: 64 input → 128 hidden (reduced from 5 layers which was excessive)
        self.lstm = nn.LSTM(64, 128, batch_first=True, num_layers=2, dropout=0.3)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        # x: (batch_size, seq_len, 64)

        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch_size, 128) — take the last layer's hidden state
        out = h_n[-1]
        # out: (batch_size, 128)

        out = self.dropout(out)
        out = self.fc1(out)
        # out: (batch_size, 2)

        return out