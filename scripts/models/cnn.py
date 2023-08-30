import torch
import torch.nn as nn

# 1-d convolutional neural network
class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, output_size, history_length, dropout):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.dropout = dropout
        self.history_length = history_length
        self.conv1d = self.make_conv1d()
        self.mlp = self.make_mlp()
        self.conv1d.apply(self.init_weights)

    def make_conv1d(self):
        conv1d_1 = nn.Conv1d(self.input_size, self.num_filters, 
                           self.kernel_size, 
                           padding=self.kernel_size // 2)
        
        # averge pooling
        avg_pool = nn.AvgPool1d(kernel_size=self.history_length)
        conv1d_2 = nn.Conv1d(self.num_filters, self.output_size, kernel_size=1)

        return nn.Sequential(conv1d_1, avg_pool, conv1d_2)
    
    def make_mlp(self):
        mlp = nn.Sequential(
            nn.Linear(self.num_filters, 128),
            nn.SELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.output_size))
        return mlp
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.conv1d(x).squeeze(2)

        return out
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

if __name__=='__main__':

    model = CNNModel(14, 64, 3, 6, 4, 0.2)
    x = torch.randn(64, 4, 14)
    out = model(x)
    print(out.size())

