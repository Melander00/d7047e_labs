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
    




    def generate_caption(self, features, vocab, max_length=20):

            result = []

            hidden = None

            # project image features
            inputs = self.featurefc(features)

            # add sequence dimension
            inputs = inputs.unsqueeze(1)

            for _ in range(max_length):

                # LSTM forward
                output, hidden = self.rec(inputs, hidden)

                # output:
                # (batch, seq_len=1, hidden)

                output = self.fc(output[:, -1, :])

                # choose best token
                predicted = output.argmax(dim=1)

                predicted_idx = predicted.item()

                word = vocab.itos[predicted_idx]

                if word == "<EOS>":
                    break

                if word not in ["<SOS>", "<PAD>"]:
                    result.append(word)

                # next input token
                inputs = self.emb(predicted)

                # add seq dimension
                inputs = inputs.unsqueeze(1)

            return " ".join(result)  