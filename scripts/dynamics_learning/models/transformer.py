import torch 
import torch.nn as nn
import math
from .mlp import MLP

class Transformer(nn.Module):

    def __init__(self, input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout, decoder_sizes, 
                 output_size, encoder_output, causal_masking, **kwargs):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout, causal_masking)
        decoder_input = history_length if encoder_output == 'output' else num_layers
        if encoder_output == 'output':
            decoder_input = encoder_dim / history_length
        else:
            decoder_input = encoder_dim
        
        self.decoder = MLP(decoder_input, history_length, decoder_sizes, output_size, dropout)
        self.encoder_output = encoder_output
        # self.decoder = nn.Linear(encoder_dim, output_size)  
        # self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, args=None):
        
        x_encoded = self.encoder(x)
        # Average mean pooling
        # x_encoded = torch.mean(x_encoded, dim=1)    
        # last hidden state
        if self.encoder_output == 'output':
            x_encoded = x_encoded[:,-1,:]

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
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)
    
class Encoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, history_length, ffn_hidden, n_layers, dropout, causal_masking):
        super(Encoder, self).__init__()

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, ffn_hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.src_mask = None
        self.causal_masking = causal_masking
        self.history_length = history_length

    def forward(self, x):

        # Transpose x such that the shape is (history_length, batch_size, input_size)        
        self.src_mask = self.generate_square_subsequent_mask(self.history_length).to(x.device)
        x = self.encoder(x)
        x = self.pos_encoder(x)
        
        if self.causal_masking:
            x = self.transformer_encoder(x, self.src_mask)
        else:
            x = self.transformer_encoder(x)

        return x
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



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
