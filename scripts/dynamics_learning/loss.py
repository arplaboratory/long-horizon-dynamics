import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, target):
        
        loss = (pred - target)**2
        loss = loss.sum(dim=1)
        loss = loss.mean(0)
        return loss