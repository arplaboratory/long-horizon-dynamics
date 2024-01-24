import torch
import torch.nn as nn
from torch.autograd import Variable
from .mlp import MLP

class LSTM(nn.Module):
  def __init__(self, input_size, encoder_sizes, num_layers, history_len, decoder_sizes, output_size, dropout,
               encoder_output, **kwargs):
    super(LSTM, self).__init__()
    self.encoder = nn.LSTM(input_size=input_size, hidden_size=encoder_sizes[0],
                           num_layers=num_layers, batch_first=True, dropout=dropout)
    decoder_input = history_len if encoder_output == 'output' else num_layers
    self.decoder = MLP(encoder_sizes[0], decoder_input, decoder_sizes, output_size, dropout)
    self.encoder_output = encoder_output
    self.num_layers = num_layers
    self.hidden_size = encoder_sizes[0]
    self.memory = None


  def forward(self, x, init_memory):
    h = self.init_memory(x.shape[0], x.device) if init_memory else self.memory
    x, h = self.encoder(x, h)
    self.memory = h
    x = self.decoder(x) if self.encoder_output == 'output' else self.decoder(h[0].permute(1, 0, 2))
    return x

  def init_memory(self, batch_size, device):
    return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device=device),
            Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device=device))