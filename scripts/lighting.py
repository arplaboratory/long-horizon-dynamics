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


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider

plt.rcParams["figure.figsize"] = (19.20, 10.80)
font = {"family" : "sans",
        "weight" : "normal",
        "size"   : 28}
matplotlib.rc("font", **font)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
colors = ["#7d7376","#365282","#e84c53","#edb120"]

OUTPUT_FEATURES = ['u', 'v', 'w',
                           'r11', 'r21', 'r31', 
                           'r12', 'r22', 'r32',
                           'r13', 'r23', 'r33',
                           'p', 'q', 'r']

class DynamicsLearning(pytorch_lightning.LightningModule):
    def __init__(self, args, resources_path, experiment_path, 
                 input_size, output_size, num_layers, sample_data, train_steps = None, 
                 valid_steps = None, pred_steps = None):
        super().__init__()
        self.args = args
        self.resources_path = resources_path
        self.experiment_path = experiment_path
        self.train_step = train_steps
        self.valid_step = valid_steps
        self.pred_step = pred_steps
        self.input_size = input_size
        self.sample_data = sample_data
        self.output_size = output_size

        if args.model_type == "mlp":

            if args.history_length > 0:
                input_size = input_size * args.history_length
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
        # else print error warning
        else:
            print("Error: Model type not found!")
            
        self.mse = torch.nn.MSELoss()
        self.frobenius = FrobeniusLoss()
        self.best_valid_loss = 1e8
        self.running_loss_sum = 0
        self.batch_count = 0

    def forward(self, x):
        
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
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

        x_current = x
        loss = 0

        common_slice = slice(self.input_size, None)
        for t in range(self.args.unroll_length):
        
            y_hat = self.forward(x_current)
            label = y[:, :-4, t].reshape(y_hat.shape)
            loss += self.mse(y_hat, label)
                
            if t < self.args.unroll_length - 1:
                u_curr = y[:, -4:, t]
                x_current = torch.cat([x[:, common_slice], y_hat, u_curr], dim=1)
            
            self.running_loss_sum += loss.item()
            self.batch_count += 1
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_epoch_loss', self.running_loss_sum / self.batch_count, prog_bar=True)

        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.float()
        y = y.float()
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return y_hat
    
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        x = x.float()
        y = y.float()

        x_current = x
        loss = 0

        common_slice = slice(self.input_size, None)
        for t in range(self.args.unroll_length):
        
            y_hat = self.forward(x_current)
            label = y[:, :-4, t].reshape(y_hat.shape)
            loss += self.mse(y_hat, label)
                
            if t < self.args.unroll_length - 1:
                u_curr = y[:, -4:, t]
                x_current = torch.cat([x[:, common_slice], y_hat, u_curr], dim=1)
            
            self.running_loss_sum += loss.item()
            self.batch_count += 1
            self.log('valid_loss', loss, prog_bar=True)
            self.log('valid_epoch_loss', self.running_loss_sum / self.batch_count, prog_bar=True)
            
        return loss

    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack(outputs).mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            print("\n\nValidation loss improved!")
            print("Best valid loss    %2.4f\n" % self.best_valid_loss)
            torch.save(self.model.state_dict(), self.experiment_path + "checkpoints/model.pth")

    def on_train_epoch_end(self):
        self.running_loss_sum = 0
        self.batch_count = 0
        self.plot_predictions()
    
    def plot_predictions(self):

        # Plot predictions and ground truth and save to pdf

        self.eval()
        preds = []
        with torch.no_grad():
            x, y = self.sample_data
            x = x.float().to(self.args.device)
            y = y.float().to(self.args.device)

            x_current = x
            common_slice = slice(self.input_size, None)
            for t in range(self.args.unroll_length):
                y_hat = self.forward(x_current)
                preds.append(y_hat)
                if t < self.args.unroll_length - 1:
                    u_curr = y[:, -4:, t]
                    x_current = torch.cat([x[:, common_slice], y_hat, u_curr], dim=1)
            
        preds = torch.stack(preds)
        self.train()

        # Plotting on pdf file
        preds = preds.cpu().numpy()
        y = y[:, :-4, :].cpu().numpy()

        preds = preds[:, 0, :]
        y = y[0, :, :].T

        with PdfPages(self.experiment_path + "plots/predictions.pdf") as pdf:
            for i in range(len(OUTPUT_FEATURES)):
                fig = plt.figure()
                plt.plot(y[:, i], label="True")
                plt.plot(preds[:, i], label="Predicted")
                plt.xlabel("Time (s)")
                plt.ylabel(OUTPUT_FEATURES[i])
                plt.legend()
                pdf.savefig(fig)
                plt.close(fig)



