import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_size, history_len, decoder_sizes, output_size, dropout, **kwargs):
    super(MLP, self).__init__()
    self.model = self.make(input_size * history_len, decoder_sizes, output_size, dropout)

  def make(self, input_size, decoder_sizes, output_size, dropout):
    layers = []
    layers.append(nn.Linear(input_size, decoder_sizes[0]))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout))
    for i in range(len(decoder_sizes) - 1):
      layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
      layers.append(nn.GELU())
      layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(decoder_sizes[-1], output_size))
    return nn.Sequential(*layers)

  def forward(self, x, args=None):
    x = x.reshape(x.shape[0], -1)
    x = self.model(x)
    return x