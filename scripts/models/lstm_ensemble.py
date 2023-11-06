import torch 
import torch.nn as nn
from torch.autograd import Variable

class LSTMEnsemble(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, ensemble_size):
        super(LSTMEnsemble, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.ensemble_size = ensemble_size #ensemble size

        self.model = nn.ModuleList([nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) for _ in range(ensemble_size)])
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):

        preds = []
        for i in range(self.ensemble_size):
            h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
            c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
            output, (hn, cn) = self.model[i](x, (h_0, c_0))
            hn = hn.view(-1, self.hidden_size)
            out = self.relu(hn)
            out = self.fc_1(out)
            out = self.relu(out)
            out = self.fc(out)
            preds.append(out)

        preds = torch.stack(preds, dim=0)
        preds = torch.mean(preds, dim=0)
    
        return preds
    
        
        

if __name__=='__main__':
    input_size = 19 #number of features
    hidden_size = 2 #number of features in hidden state
    num_layers = 1 #number of stacked lstm layers

    num_classes = 15 #number of output classes 
    x = torch.randn(64, 4, 19)
    model = LSTMEnsemble(num_classes, input_size, hidden_size, num_layers, x.shape[1], 5)

    out = model(x)
    print(out.size())