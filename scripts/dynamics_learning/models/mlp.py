import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.model = self.make()
        self.model.apply(self.init_weights)

    def make(self):
        layers = nn.ModuleList()
        for i in range(len(self.num_layers) - 1):
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.num_layers[i]))
                layers.append(nn.SELU())
                # batch normalization
                layers.append(nn.Dropout(self.dropout))
            else:
                layers.append(nn.Linear(self.num_layers[i-1], self.num_layers[i]))
                layers.append(nn.SELU())
                layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.num_layers[i], self.output_size))
        layers = nn.Sequential(*layers)
        return layers
    
    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


if __name__=='__main__':
    input_size = 19 #number of features
    output_size = 256
    num_layers = [256, 256, 512]
    dropout = 0.2

    model = MLP(input_size, output_size, num_layers, dropout)

    x = torch.randn(16384, 19)

    y = model(x)

    print(y.shape)
        
