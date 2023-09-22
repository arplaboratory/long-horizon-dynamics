import sys
import os
import time
import warnings
import torch
import pytorch_lightning

from config import parse_args
from models.mlp import MLP
from models.lstm import LSTM
from models.cnn import CNNModel
from models.tcn import TCN

from loss import FrobeniusLoss
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

        if args.model_type == "mlp":
            self.model = MLP(input_size=input_size, 
                    output_size=output_size,
                    num_layers=num_layers, 
                    dropout=args.dropout)
        elif args.model_type == "lstm":
            self.model = LSTM(input_size=input_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              output_size=output_size,
                              history_length=args.history_length,
                              dropout=args.dropout)
        elif args.model_type == "cnn":
            self.model = CNNModel(input_size=input_size,
                                  num_filters=args.num_filters,
                                  kernel_size=args.kernel_size,
                                  output_size=output_size,
                                  history_length=args.history_length,
                                  num_layers=args.num_layers,
                                  residual=args.residual,
                                  dropout=args.dropout)
        elif args.model_type == "tcn":
            self.model = TCN(num_inputs=input_size,
                             num_channels=args.num_channels,
                             kernel_size=args.kernel_size,
                             dropout=args.dropout,
                             num_outputs=input_size-4)
            
        self.mse = torch.nn.MSELoss()
        self.frobenius = FrobeniusLoss()
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

        # return {
        #     'optimizer': optimizer,
        # }
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.float()
        y = y.float()
        y_hat = self.model(x)
        
        # Velocity
        pred_vel = y_hat[:, :3]

        # Attitude
        if self.args.attitude == 'euler':
            pred_att = y_hat[:, 3:6]
            pred_ang_vel = y_hat[:, 6:9]

            gt_att = y[:, 3:6]
            gt_ang_vel = y[:, 6:9]

            loss = self.mse(pred_vel, y[:, :3]) + self.mse(pred_att, gt_att) + self.mse(pred_ang_vel, gt_ang_vel)

        elif self.args.attitude == 'quaternion':
            pred_att = y_hat[:, 3:7]
            pred_ang_vel = y_hat[:, 7:10]

            gt_att = y[:, 3:7]
            gt_ang_vel = y[:, 7:10]

            loss = self.mse(pred_vel, y[:, :3]) + self.mse(pred_att, gt_att) + self.mse(pred_ang_vel, gt_ang_vel)

        elif self.args.attitude == 'rotation':
            pred_att = y_hat[:, 3:12].reshape(-1, 3, 3)
            pred_ang_vel = y_hat[:, 12:15]

            gt_att = y[:, 3:12].reshape(-1, 3, 3)
            gt_ang_vel = y[:, 12:15]

            loss_vel = self.mse(pred_vel, y[:, :3])

            if self.args.rot_loss:
                loss_att = self.frobenius(pred_att, gt_att)
            else:
                loss_att = self.mse(pred_att, gt_att)
            
            loss_ang_vel = self.mse(pred_ang_vel, gt_ang_vel)

            loss = self.args.vel_loss * loss_vel + self.args.att_loss * loss_att + self.args.ang_vel_loss * loss_ang_vel        

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
        
        
        # Velocity
        pred_vel = y_hat[:, :3]

        # Attitude
        if self.args.attitude == 'euler':
            pred_att = y_hat[:, 3:6]
            pred_ang_vel = y_hat[:, 6:9]

            gt_att = y[:, 3:6]
            gt_ang_vel = y[:, 6:9]

            loss = self.mse(pred_vel, y[:, :3]) + self.mse(pred_att, gt_att) + self.mse(pred_ang_vel, gt_ang_vel)

        elif self.args.attitude == 'quaternion':
            pred_att = y_hat[:, 3:7]
            pred_ang_vel = y_hat[:, 7:10]

            gt_att = y[:, 3:7]
            gt_ang_vel = y[:, 7:10]

            loss = self.mse(pred_vel, y[:, :3]) + self.mse(pred_att, gt_att) + self.mse(pred_ang_vel, gt_ang_vel)

        elif self.args.attitude == 'rotation':
            pred_att = y_hat[:, 3:12].reshape(-1, 3, 3)
            pred_ang_vel = y_hat[:, 12:15]

            gt_att = y[:, 3:12].reshape(-1, 3, 3)
            gt_ang_vel = y[:, 12:15]

            loss_vel = self.mse(pred_vel, y[:, :3])

            if self.args.rot_loss:
                loss_att = self.frobenius(pred_att, gt_att)
            else:
                loss_att = self.mse(pred_att, gt_att)
            
            loss_ang_vel = self.mse(pred_ang_vel, gt_ang_vel)

            loss = self.args.vel_loss * loss_vel + self.args.att_loss * loss_att + self.args.ang_vel_loss * loss_ang_vel        

        self.log('valid_loss', loss, prog_bar=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack(outputs).mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            print("\n\nValidation loss improved!")
            print("Best valid loss    %2.4f\n" % self.best_valid_loss)
            torch.save(self.model.state_dict(), self.experiment_path + "checkpoints/model.pth")

    

    

