import torch
import torch.nn as nn

class FrobeniusLoss(nn.Module):
    def __init__(self):
        super(FrobeniusLoss, self).__init__()

    def forward(self, predicted_rotation, target_rotation):
        # Compute the loss as the Frobenius norm of the difference
        # between the transpose of predicted rotation and the identity matrix
        loss = torch.norm(torch.matmul(predicted_rotation.transpose(1, 2), target_rotation) - torch.eye(3).to(predicted_rotation.device), 'fro')
        
        return loss
    
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, target):
        
        loss = (pred - target)**2
        loss = loss.sum(dim=1)
        loss = loss.mean(0)

        return loss