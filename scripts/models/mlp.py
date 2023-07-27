import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, input_size, num_layers, dropout):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.model = self.make()

    def make(self):
        layers = nn.ModuleList()
        for i in range(len(self.num_layers) - 1):
            if i == 0:
                layers.append(nn.Linear(self.input_size, self.num_layers[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))

            else:
                layers.append(nn.Linear(self.num_layers[i], self.num_layers[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.num_layers[i + 1], self.input_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        layers = nn.Sequential(*layers)
        return layers
    
    def forward(self, x):
        return self.model(x)
        
