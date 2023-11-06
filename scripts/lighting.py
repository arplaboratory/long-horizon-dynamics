import sys
import os
import time
import warnings
import torch
import pytorch_lightning
import copy

from config import parse_args
from models.mlp import MLP
from models.lstm import LSTM
from models.cnn import CNNModel
from models.tcn import TCN
from models.lstm_ensemble import LSTMEnsemble
from models.tcn_ensemble import TCNEnsemble

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
plt.switch_backend('agg')

colors = ["#7d7376","#365282","#e84c53","#edb120"]



OUTPUT_FEATURES = {
    "euler": ["u", "v", "w", "phi", "theta", "psi", "p", "q", "r"],
    "quaternion": ["u", "v", "w", "q0", "q1", "q2", "q3", "p", "q", "r"],
    "rotation": ["u", "v", "w", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "p", "q", "r"]
}

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
            self.model = LSTM(num_classes=output_size,
                              input_size=input_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              seq_length=args.history_length)
            
        elif args.model_type == "lstm_ensemble":
            self.model = LSTMEnsemble(num_classes=output_size,
                                      input_size=input_size,
                                      hidden_size=args.hidden_size,
                                      num_layers=args.num_layers,
                                      seq_length=args.history_length,
                                      ensemble_size=args.ensemble_size)
            
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
            
        elif args.model_type == "tcn_ensemble":
            self.model = TCNEnsemble(num_inputs=input_size,
                                     num_channels=args.num_channels,
                                     kernel_size=args.kernel_size,
                                     dropout=args.dropout,
                                     num_outputs=input_size-4,
                                     ensemble_size=args.ensemble_size)
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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.float()
        y = y.float()

        x_current = x
        batch_loss = 0.0

        # common_slice = slice(self.input_size, None)
        for t in range(self.args.unroll_length):
            y_hat = self.forward(x_current)

            label = y[:, :-4, t].reshape(y_hat.shape)
            batch_loss += self.mse(y_hat, label)
                
            if t < self.args.unroll_length - 1:
                u_curr = y[:, -4:, t]
                x_current = torch.cat([x_current[:, 1:, :], torch.cat([y_hat, u_curr], dim=1).unsqueeze(dim=1)], dim=1)
            
        self.running_loss_sum += batch_loss.item() / self.args.unroll_length
        self.batch_count += 1
        self.log('train_loss', batch_loss, prog_bar=True)
        self.log('train_epoch_loss', self.running_loss_sum / self.batch_count, prog_bar=True)
            
        del y_hat, label

        return batch_loss
    
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
        batch_loss = 0.0

        # common_slice = slice(self.input_size, None)
        for t in range(self.args.unroll_length):
            y_hat = self.forward(x_current)

            label = y[:, :-4, t].reshape(y_hat.shape)
            batch_loss += self.mse(y_hat, label)
                
            if t < self.args.unroll_length - 1:
                u_curr = y[:, -4:, t]
                x_current = torch.cat([x_current[:, 1:, :], torch.cat([y_hat, u_curr], dim=1).unsqueeze(dim=1)], dim=1)
            
        self.running_loss_sum += batch_loss.item() / self.args.unroll_length
        self.batch_count += 1
        self.log('valid_loss', batch_loss, prog_bar=True)
        self.log('valid_epoch_loss', self.running_loss_sum / self.batch_count, prog_bar=True)

        del y_hat, label
        
            
        return batch_loss

    def validation_epoch_end(self, outputs):
        valid_loss = torch.stack(outputs).mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            print("\n\nValidation loss improved!")
            print("Best valid loss    %2.4f\n" % self.best_valid_loss)
            torch.save(self.model.state_dict(), self.experiment_path + "checkpoints/model.pth")

        # Release memory after evaluation
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        self.running_loss_sum = 0
        self.batch_count = 0

        # plot every 10 epochs
        if self.current_epoch % self.args.plot_freq == 0:
            self.plot_predictions()

        torch.cuda.empty_cache()

    def plot_predictions(self):

        # Plot predictions and ground truth and save to pdf
        fig, axes = copy.deepcopy(self.fig), copy.deepcopy(self.axes)

        self.eval()
        preds = []

        with torch.no_grad():

            x_current = self.sample_inputs
            y = self.sample_labels

            # common_slice = slice(self.input_size, None)
            for t in range(self.args.unroll_length):
                
            
                y_hat = self.forward(x_current)

                preds.append(y_hat)

                if t < self.args.unroll_length - 1:
                    u_curr = y[:, -4:, t]
                    x_current = torch.cat([x_current[:, 1:, :], torch.cat([y_hat, u_curr], dim=1).unsqueeze(dim=1)], dim=1)
            
                # Release y_hat from memory
                del y_hat

        preds = torch.cat(preds, dim=0)
        self.train()

        # Plot pred states with different colors
        preds = preds.cpu().numpy()

        for i in range(preds.shape[1]):
            row = i // 3
            col = i % 3

            ax = axes[row, col]

            # Make sure while plotting the predictions, the history is not plotted and the prediciton are shifted to the right by history length
            ax.plot(range(self.args.history_length, self.args.history_length + self.args.unroll_length), preds[:, i], color=colors[3])
                    

        plt.tight_layout()  # Adjust subplot layout
        plt.savefig(self.experiment_path + "plots/predictions.png")
        plt.close()
    
    def on_train_start(self):

        # Plot the history of states and unrolled labels and save to png. Plots must be aesthetically pleasing

        fig, axes = plt.subplots(5, 3, figsize=(20, 20))
        fig.suptitle("Input and Label States", fontsize=16)

        fig.subplots_adjust(hspace=0.8, wspace=0.8)

        x, y = self.sample_data

        input_states = x[0].reshape(self.args.history_length, -1)[:, :-4]
        label_states = y[0][:-4, :].T

        # Concatenate the input and label states
        states = torch.cat((input_states, label_states), dim=0)


        time_values = [i for i in range(states.shape[0])]

        for i in range(self.output_size):
            row = i // 3
            col = i % 3

            ax = axes[row, col]

            # Make sure the history and labels are plotted with different colors
            ax.plot(time_values[:self.args.history_length+1], states[:self.args.history_length+1, i], color=colors[2])
            ax.plot(time_values[self.args.history_length:], states[self.args.history_length:, i], color=colors[1])

            # ax.plot(states[:, i], color=colors[2])

            # Add y-axis labels for all subplots
            print(OUTPUT_FEATURES[self.args.attitude])
            ax.set_ylabel(OUTPUT_FEATURES[self.args.attitude][i])

            if row == 4:
                ax.set_xlabel("Time (s)")

            # Calculate the range for the y-axis based on the minimum and maximum values
            min_value = states[:, i].min().item()
            max_value = states[:, i].max().item()
            y_range = max_value - min_value
            padding = 0.8 * y_range  # 30% padding

            # Set y-axis limits with padding
            ax.set_ylim(min_value - padding, max_value + padding)

        # Create two common legends with different labels and colors
        legend_history =    plt.Line2D([0], [0], color=colors[2], label='History')
        legend_gt_rollout = plt.Line2D([0], [0], color=colors[1], label='Ground Truth Rollout')
        legend_prediction = plt.Line2D([0], [0], color=colors[3], label='Prediction')

        # Add the legends to the figure
        fig.legend(handles=[legend_history, legend_gt_rollout, legend_prediction], loc="upper right")

        # Adjust the legend's position relative to the rest of the plot
        fig.canvas.draw()

        # Set x-axis labels for all subplots
        for i in range(5):
            for j in range(3):
                axes[i, j].set_xticks(range(0, len(time_values), len(time_values)//5))
                axes[i, j].set_xticklabels(range(0, len(time_values), len(time_values)//5))

        plt.tight_layout()  # Adjust subplot layout

        self.fig, self.axes = fig, axes
        self.sample_inputs = self.sample_data[0].float().to(self.args.device)[0:1,]
        self.sample_labels = self.sample_data[1].float().to(self.args.device)[0:1,]



    
