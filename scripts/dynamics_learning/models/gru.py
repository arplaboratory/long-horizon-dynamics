import torch
import torch.nn as nn

class GRU(nn.Module):
  def __init__(self, input_size, encoder_dim, encoder_sizes, num_layers, history_length, dropout,
               output_type):
    super(GRU, self).__init__()
    self.encoder = nn.GRU(input_size=input_size, hidden_size=encoder_sizes,
                          num_layers=num_layers, batch_first=True, dropout=dropout)
    encoder_sizes *= num_layers if output_type == 'hidden' else history_length
    self.linear = nn.Linear(encoder_sizes, encoder_dim)
    self.hidden_size = encoder_sizes
    self.num_layers = num_layers
    self.history_length = history_length
    self.output_type = output_type

  def forward(self, x):
    batch_size = x.shape[0]
    x, h = self.encoder(x)
    x = h.permute(1, 0, 2).reshape(batch_size, -1) 
    x = self.linear(x)
    return x
  
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