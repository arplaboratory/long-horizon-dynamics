import torch 
import torch.nn as nn
import math
from .mlp import MLP

class Transformer(nn.Module):

    def __init__(self, input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout, decoder_sizes, output_size, **kwargs):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout)
        self.decoder = MLP(encoder_dim / history_length, history_length, decoder_sizes, output_size, dropout)
        
    def forward(self, x, args=None):

        x_encoded = self.encoder(x)
        # Average mean pooling
        x_encoded = torch.mean(x_encoded, dim=1)    
        # last hidden state
        #x_encoded = x_encoded[:, -1, :]
        x_decoded = self.decoder(x_encoded)
        return x_decoded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Encoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, history_length, ffn_hidden, n_layers, dropout):
        super(Encoder, self).__init__()

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, ffn_hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)

        # Implement causal masking
        mask = torch.tril(torch.ones(x.size(1), x.size(1)))
        mask = mask.unsqueeze(0).expand(x.size(0), -1, -1)
        mask = mask.to(x.device)
        x = self.transformer_encoder(x)
        return x



if __name__=='__main__':
    input_size = 19 #number of features
    output_dim = 256
    num_heads = 8
    history_length = 10
    ffn_hidden = 512
    num_layers = 2
    dropout = 0.2

    model = Transformer(input_size, output_dim, num_heads, history_length, ffn_hidden, num_layers, dropout)

    x = torch.randn(32, 10, 19)

    y = model(x)

    print(y.shape)