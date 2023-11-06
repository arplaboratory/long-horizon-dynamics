import torch
import torch.nn as nn

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

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

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
    
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, num_outputs):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], num_outputs)  # Output shape matches (batch_size, num_features-4)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Transpose input to (batch_size, num_features, history_length)
        y = self.tcn(x)
        y = self.fc(y[:, :, -1])  # Take the output of the last time step
        return y

class TCNEnsemble(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, num_outputs, ensemble_size):
        super(TCNEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        
        self.models = nn.ModuleList([TCN(num_inputs, num_channels, kernel_size, dropout, num_outputs) for _ in range(ensemble_size)])



    def forward(self, x):
        
        preds = []
        
        for i in range(self.ensemble_size):

            output = self.models[i](x)
            preds.append(output)

        preds = torch.stack(preds, dim=0)
        preds = torch.mean(preds, dim=0)
        return preds
           
# Example usage:
if __name__ == '__main__':
    batch_size = 32
    history_length = 100
    num_features = 19
    num_channels = [64, 128, 256]
    kernel_size = 3
    dropout = 0.2
    num_outputs = num_features - 4  # Set the number of output features
    ensemble_size = 5

    # Create a random input tensor with the shape (batch_size, history_length, num_features)
    input_tensor = torch.randn(batch_size, history_length, num_features)

    # Create the TCN model
    tcn = TCNEnsemble(num_features, num_channels, kernel_size, dropout, num_outputs, ensemble_size)

    # Forward pass
    output = tcn(input_tensor)
    print("Output shape:", output.shape)
