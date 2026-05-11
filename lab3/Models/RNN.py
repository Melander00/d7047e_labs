import torch
import torch.nn as nn


class CaptionRNN(nn.Module):
    def __init__(self,vocab_size:int,):
        super().__init__()
        embed=256
        hidden_size=512

        
        self.featurefc=nn.Linear(128,embed)

        self.emb=nn.Embedding(embedding_dim=embed,num_embeddings=vocab_size
                     )

        
        
        self.rec=nn.LSTM(
            input_size=embed,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,

        )

        self.fc=nn.Linear(hidden_size,vocab_size)

        # Define layers here
        # Example:
        # self.layer = nn.Linear(in_features, out_features)

    def forward(self, features, captions):

    # image features
        features = self.featurefc(features)

        # caption embeddings
        embeddings = self.emb(captions)

        # add image feature as first timestep
        features = features.unsqueeze(1)

        # concatenate
        inputs = torch.cat((features, embeddings), dim=1)

        # LSTM
        outputs, _ = self.rec(inputs)

        # vocab scores
        outputs = self.fc(outputs)
        

        return outputs