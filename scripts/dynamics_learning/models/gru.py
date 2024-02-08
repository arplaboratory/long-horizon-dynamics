import torch
import torch.nn as nn
from torch.autograd import Variable
from .mlp import MLP

class GRU(nn.Module):
  def __init__(self, input_size, encoder_sizes, num_layers, history_len, decoder_sizes, output_size, dropout,
               encoder_output, **kwargs):
    super(GRU, self).__init__()
    self.encoder = nn.GRU(input_size=input_size, hidden_size=encoder_sizes[0],
                          num_layers=num_layers, batch_first=True, dropout=dropout)
    decoder_input = history_len if encoder_output == 'output' else num_layers
    if encoder_output == 'hidden':
      decoder_input = 2 * (encoder_sizes[0] / history_len)
    elif encoder_output == 'output':
      decoder_input = encoder_sizes[0] / history_len
    
    else:
      decoder_input = encoder_sizes[0]

    self.decoder = MLP(decoder_input, history_len, decoder_sizes, output_size, dropout)
    self.dropout = nn.Dropout(dropout)
    self.encoder_output = encoder_output
    self.num_layers = num_layers
    self.hidden_size = encoder_sizes[0]
    self.memory = None

  def forward(self, x, init_memory):
    h = self.init_memory(x.shape[0], x.device) if init_memory else self.memory
    x, h = self.encoder(x, h)
    self.memory = h


    if self.encoder_output == 'hidden':
      x_encoder = torch.cat([h[0][-1], h[1][-1]], dim=1)

    elif self.encoder_output == 'output':
      x_encoder = x[:, -1, :]

    else:
      x_encoder = x
      
    x_encoder = self.dropout(x_encoder)

    x = self.decoder(x_encoder)
    
    return x

  def init_memory(self, batch_size, device):
    return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device=device)
  
if __name__=='__main__':
    input_size = 19 #number of features
    encoder_sizes = 256
    num_layers = 2
    history_len = 10
    dropout = 0.2
    output_type = 'hidden'
    encoder_dim = 256

    model = GRU(input_size, encoder_dim, encoder_sizes, num_layers, history_len, dropout, output_type)

    x = torch.randn(16384, 10, 19)

    y = model(x)

    print(y.shape)