import torch 
import torch.nn as nn
import math
from .mlp import MLP
from torch.nn import functional as F


class Transformer(nn.Module):

    def __init__(self, input_size, encoder_dim, num_heads, history_length, ffn_hidden, num_layers, dropout, decoder_sizes, 
                 output_size, encoder_output, causal_masking, pos_encoder_type, decoder_type, **kwargs):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_size, encoder_dim, num_heads, history_length, ffn_hidden, 
                               num_layers, dropout, causal_masking, pos_encoder_type)
        decoder_input = history_length if encoder_output == 'output' else num_layers
        if encoder_output == 'output':
            decoder_input = encoder_dim / history_length
        else:
            decoder_input = encoder_dim
        
        if decoder_type == 'mlp':
            self.decoder = MLP(decoder_input, history_length, decoder_sizes, output_size, dropout)
        elif decoder_type == 'linear':
            self.decoder = nn.Linear(encoder_dim * history_length, output_size)

        self.encoder_output = encoder_output
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
            x_encoded = x_encoded.reshape(x_encoded.shape[0], -1)
            
        x_decoded = self.decoder(x_encoded)
        return x_decoded

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(FixedPositionalEncoding, self).__init__()
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
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.positional_encoding, -0.1, 0.1)

    def forward(self, x):

        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.positional_encoding[:x.size(0), :]
        return self.dropout(x)
    
def get_pos_encoder(pos_encoder_type):
    if pos_encoder_type == 'learnable':
        return LearnablePositionalEncoding
    elif pos_encoder_type == 'fixed':
        return FixedPositionalEncoding
    else:
        raise ValueError(f"Positional encoder type {pos_encoder_type} not supported")
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))
    
class Encoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, history_length, ffn_hidden, 
                 n_layers, dropout, causal_masking, pos_encoder_type):
        super(Encoder, self).__init__()

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = get_pos_encoder(pos_encoder_type)(d_model, dropout, history_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, ffn_hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.src_mask = None
        self.causal_masking = causal_masking
        self.history_length = history_length
        self.d_model = d_model

        self.act = _get_activation_fn("gelu")
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):

        # Transpose x such that the shape is (history_length, batch_size, input_size)
        x = x.transpose(0, 1)
                
        self.src_mask = self.generate_square_subsequent_mask(self.history_length).to(x.device)
        x = self.encoder(x) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        x = self.pos_encoder(x)
        
        if self.causal_masking:
            x = self.transformer_encoder(x, self.src_mask)
        else:
            x = self.transformer_encoder(x)

        x = self.act(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_length, d_model]
        x = self.dropout1(x)
        
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
