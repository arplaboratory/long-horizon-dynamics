import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class MLP(nn.Module):
  def __init__(self, input_size, history_len, decoder_sizes, output_size, dropout, **kwargs):
    super(MLP, self).__init__()
    self.model = self.make(input_size, decoder_sizes, output_size, dropout)

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
  

class TCN(nn.Module):
    def __init__(self, input_size, encoder_sizes, history_len, decoder_sizes, output_size, kernel_size, dropout, **kwargs):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, encoder_sizes, kernel_size=kernel_size, dropout=dropout)
        self.decoder = MLP(encoder_sizes[-1], history_len, decoder_sizes, output_size, dropout)

    def forward(self, x,  args=None):
        x = x.permute(0, 2, 1)  # Transpose input to (batch_size, num_features, history_length)
        y = self.tcn(x)
        y = y[:, :, -1]
        y = self.decoder(y) # Take the output of the last time step
        return y
    
# class TCN(nn.Module):
#   def __init__(self, input_size, encoder_sizes, history_len, decoder_sizes, output_size, kernel_size, dropout, **kwargs):
#     super(TCN, self).__init__()
#     encoder_sizes.insert(0, input_size)
#     self.encoder = nn.Sequential(*[ResidualTemporalBlock(encoder_sizes[i - 1], encoder_sizes[i],
#                                                          kernel_size=kernel_size, stride=1, dilation=2 ** i,
#                                                          padding=(kernel_size - 1) * (2 ** i), dropout=dropout)
#                                    for i in range(1, len(encoder_sizes))])
#     self.decoder = MLP(encoder_sizes[-1], history_len, decoder_sizes, output_size, dropout)
#     self.history_len = history_len

#   def forward(self, x, args=None):
#     x = x.permute(0, 2, 1)
#     x = self.encoder(x)
#     # x = x[:, :, -1:]
#     x = self.decoder(x)
#     return x

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

#----------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()