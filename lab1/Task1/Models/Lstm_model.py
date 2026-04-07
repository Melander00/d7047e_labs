import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        print("LSTM constucted")
        # define layers here later
        self.embedding = nn.Embedding(vocab_size, 25)
        self.lstm=nn.LSTM(25,64,batch_first=True)
        self.relu1=nn.ReLU()
        self.fc1=nn.Linear(64,2)

    def forward(self, x):
        
        x = self.embedding(x)
        # → (batch_size, seq_len, embedding_dim)

        out, _ = self.lstm(x)
        # → (batch_size, seq_len, hidden_size)

        out = out[:, -1, :]
        # → (batch_size, hidden_size)
        out = self.relu1(out)
        out = self.fc1(out)
        # → (batch_size, 2)

        return out
        
        
        
        
        
        
        
      