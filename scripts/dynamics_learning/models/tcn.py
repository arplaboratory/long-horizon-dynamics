import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .mlp import MLP

class TCN(nn.Module):
  def __init__(self, input_size, encoder_sizes, history_len, decoder_sizes, output_size, kernel_size, dropout, **kwargs):
    super(TCN, self).__init__()
    encoder_sizes.insert(0, input_size)
    self.encoder = nn.Sequential(*[ResidualTemporalBlock(encoder_sizes[i - 1], encoder_sizes[i],
                                                         kernel_size=kernel_size, stride=1, dilation=2 ** i,
                                                         padding=(kernel_size - 1) * (2 ** i), dropout=dropout)
                                   for i in range(1, len(encoder_sizes))])
    self.decoder = MLP(encoder_sizes[-1], history_len, decoder_sizes, output_size, dropout)
    self.history_len = history_len

  def forward(self, x, args=None):
    x = x.permute(0, 2, 1)
    x = self.encoder(x)
    # x = x[:, :, -1:]
    x = self.decoder(x)
    return x

#----------------------------------------------------------------------------

class ResidualTemporalBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.0):
    super(ResidualTemporalBlock, self).__init__()
    self.network = nn.Sequential(*[weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                                         stride=stride, padding=padding, dilation=dilation)),
                                   Chomp1d(padding),
                                   nn.GELU(),
                                   nn.Dropout(dropout),
                                   weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                                         stride=stride, padding=padding, dilation=dilation)),
                                   Chomp1d(padding),
                                   nn.GELU(),
                                   nn.Dropout(dropout)])
    self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    self.gelu = nn.GELU()

  def forward(self, x):
    out = self.network(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.gelu(out + res)

#----------------------------------------------------------------------------

class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super(Chomp1d, self).__init__()
    self.chomp_size = chomp_size

  def forward(self, x):
    return x[:, :, :-self.chomp_size].contiguous()