import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, history_length, dropout):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.history_length = history_length
        
        self.lstm = self.make_lstm()
        self.mlp = self.make_mlp()
        # self.lstm.apply(self.init_weights)
    
    def make_lstm(self):
        lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        return lstm

    def make_mlp(self):
        mlp = nn.Sequential(
            nn.Linear(self.hidden_size * self.history_length, 128),
            nn.SELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.output_size)
        )
        return mlp
    
    def forward(self, x):

        out, _ = self.lstm(x)
        
        # Flatten the LSTM output for each batch
        out = out.reshape(out.size(0), -1)
        out = self.mlp(out)

        return out
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.01)

if __name__=='__main__':
    model = LSTM(14, 128, 2, 6, 4, 0.5)
    x = torch.randn(64, 4, 14)
    out = model(x)
    print(out.size())
    