import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size):
        
        super().__init__()
        print("LSTM constucted")
        # define layers here later
        self.embedding = nn.Embedding(vocab_size, 2,padding_idx=0)
        self.lstm=nn.LSTM(2,8,batch_first=True,num_layers=1)
        
        self.dropout = nn.Dropout(0.3)
        #self.relu1=nn.ReLU()
        
        self.fc1=nn.Linear(8,2)

    def forward(self, x):
        
        x = self.embedding(x)
        # → (batch_size, seq_len, embedding_dim)

        #out, _ = self.lstm(x)
        # → (batch_size, seq_len, hidden_size)

        #out = out[:, -1, :]


        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        
        # → (batch_size, hidden_size)
        out=self.dropout(out)
       # out = self.relu1(out)
        
        out = self.fc1(out)
        # → (batch_size, 2)

        return out
        
        
        
        
        
        
        
      