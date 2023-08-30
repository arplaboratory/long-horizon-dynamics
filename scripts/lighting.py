import sys
import os
import time
import warnings
import torch
import pytorch_lightning

from config import parse_args
from models.mlp import MLP

warnings.filterwarnings("ignore")


class DynamicsLearning(pytorch_lightning.LightningModule):
    def __init__(self, args, resources_path, experiment_path, 
                 input_size, output_size, num_layers, train_steps = None, 
                 valid_steps = None, pred_steps = None):
        super().__init__()
        self.args = args
        self.resources_path = resources_path
        self.experiment_path = experiment_path
        self.train_step = train_steps
        self.valid_step = valid_steps
        self.pred_step = pred_steps
        self.model = MLP(input_size=input_size, 
                output_size=output_size,
                num_layers=num_layers, 
                dropout=args.dropout)
        self.criterion = torch.nn.MSELoss()
        self.best_valid_loss = 1e8
      

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
            
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode='min',
                    factor=0.5,
                    patience=10,
                    verbose=True
        )

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'valid_loss'
       }

        return {
            'optimizer': optimizer,
        }
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.float()
        y = y.float()

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.float()
        y = y.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return y_hat
    
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        x = x.float()
        y = y.float()
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack(outputs).mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            print("\n\nValidation loss improved!")
            print("Best valid loss    %2.4f\n" % self.best_valid_loss)
            torch.save(self.model.state_dict(), self.experiment_path + "checkpoints/model.pth")

    

    

