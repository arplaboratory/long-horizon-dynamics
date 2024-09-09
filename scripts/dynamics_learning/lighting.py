import warnings
import torch
import pytorch_lightning
import numpy as np
from .utils import quaternion_difference, quaternion_log

from .loss import MSE
from .registry import get_model

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (12,8)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # Choose your serif font here
    "font.size": 28,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

colors = ["#7d7376","#365282","#e84c53","#edb120"]
markers = ['o', 's', '^', 'D', 'v', 'p']
line_styles = ['-', '--', '-.', ':', '-', '--']

class DynamicsLearning(pytorch_lightning.LightningModule):
    def __init__(self, args, resources_path, experiment_path, 
                 input_size, output_size, max_iterations):
        super().__init__()
        self.args = args
        self.resources_path = resources_path
        self.experiment_path = experiment_path
        self.input_size = input_size
        self.output_size = output_size
        self.max_iterations = max_iterations

        # Optimizer parameters
        self.warmup_lr = args.warmup_lr
        self.cosine_lr = args.cosine_lr
        self.warmup_steps = args.warmup_steps
        self.cosine_steps = args.cosine_steps
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.adam_eps = args.adam_eps
        self.weight_decay = args.weight_decay

        # Get encoder and decoder
        self.model = get_model(args, input_size, output_size)
            
        self.loss_fn = MSE()
        
        self.best_valid_loss = 1e8
        self.verbose = False

        self.validation_step_outputs = []
        self.compounding_error = []
        self.plot_error = []

    def forward(self, x, init_memory):

        y_hat = self.model(x, init_memory) 

        if self.args.delta == True:
            if self.args.predictor_type == "velocity":
                y_hat[:, :3] = y_hat[:, :3] + x[:, -1, :3]
                y_hat[:, 3:] = y_hat[:, 3:] + x[:, -1, 7:10]            
            elif self.args.predictor_type == "attitude":
                y_hat = self.quaternion_product(y_hat, x[:, -1, 3:7])

        return y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps,
                                    lr=self.warmup_lr, weight_decay=self.weight_decay)
        schedulers = [torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=self.warmup_steps),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_steps, eta_min=self.cosine_lr),
                    torch.optim.lr_scheduler.ConstantLR(optimizer, factor=self.cosine_lr / self.warmup_lr, total_iters=self.max_iterations)]
        milestones = [self.warmup_steps, self.warmup_steps + self.cosine_steps]
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=schedulers, milestones=milestones)
        return ([optimizer],
                [{'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1}])
        
    def unroll_step(self, batch):

        x, y = batch
        x = x.float()
        y = y.float()

        x_curr = x 
        preds = []

        batch_loss = 0.0
        for i in range(self.args.unroll_length):
            y_hat = self.forward(x_curr, init_memory=True if i == 0 else False)

            # If predictor 
            if self.args.predictor_type == "velocity":
                linear_velocity_gt =  y[:, i, :3]
                angular_velocity_gt = y[:, i, 7:10]
                velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=1)
                loss = self.loss_fn(y_hat, velocity_gt)

            elif self.args.predictor_type == "attitude":
                y_hat = y_hat / torch.norm(y_hat, dim=1, keepdim=True)   # Normalize the quaternion
                attitude_gt = y[:, i, 3:7]
                loss = self.loss_fn(y_hat, attitude_gt)
            
            batch_loss += loss / self.args.unroll_length

            if i < self.args.unroll_length - 1:
                u_gt = y[:, i, -4:]

                if self.args.predictor_type == "velocity":
                    linear_velocity_pred = y_hat[:, :3]
                    angular_velocity_pred = y_hat[:, 3:]
                    attitude_gt = y[:, i, 3:7]

                    if self.args.augment_input == "va":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, va_gt, u_gt), dim=1)
                    elif self.args.augment_input == "w":
                        wind_gt = y[:, i, 10:12]
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, wind_gt, u_gt), dim=1)
                    elif self.args.augment_input == "P":
                        diff_pressure_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, diff_pressure_gt, u_gt), dim=1)
                    elif self.args.augment_input == "vawP":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        wind_gt = y[:, i, 11:13]
                        diff_pressure_gt = y[:, i, 13].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, va_gt, wind_gt, diff_pressure_gt, u_gt), dim=1)
                    else:
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)
                    # Update x_curr
                    # x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)

                elif self.args.predictor_type == "attitude":
                    linear_velocity_gt = y[:, i, :3]
                    angular_velocity_gt = y[:, i, 7:10]
                    
                    if self.args.augment_input == "va":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, va_gt, u_gt), dim=1)
                    elif self.args.augment_input == "w":
                        wind_gt = y[:, i, 10:12]
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, wind_gt, u_gt), dim=1)
                    elif self.args.augment_input == "P":
                        diff_pressure_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, diff_pressure_gt, u_gt), dim=1)
                    elif self.args.augment_input == "vawP":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        wind_gt = y[:, i, 11:13]
                        diff_pressure_gt = y[:, i, 13].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, va_gt, wind_gt, diff_pressure_gt, u_gt), dim=1)
                    else:
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)

                    # x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)
                
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)
            preds.append(y_hat)

        return preds, batch_loss
    
    def eval_trajectory(self, test_batch):

        x, y = test_batch
        x = x.float()
        y = y.float()

        x_curr = x 

        batch_loss = 0.0

        compounding_error = []
        trajectory_error_avg_unroll = []

        for i in range(self.args.unroll_length):
            y_hat = self.forward(x_curr, init_memory=True if i == 0 else False)

            if self.args.predictor_type == "velocity":
                linear_velocity_gt =  y[:, i, :3]
                angular_velocity_gt = y[:, i, 7:10]
                velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=1)
                # abs_error[i+1] = torch.mean(torch.abs(y_hat - velocity_gt), dim=0)
                loss = self.loss_fn(y_hat, velocity_gt)

                # plot loss for each sample. Compute average loss for each sample over unroll length. 
                trajectory_error_avg_unroll.append(self.loss_fn(y_hat, velocity_gt, mean=False).detach().cpu().numpy().reshape(-1, 1))

            elif self.args.predictor_type == "attitude":
                y_hat = y_hat / torch.norm(y_hat, dim=1, keepdim=True)
                attitude_gt = y[:, i, 3:7]
                q_error = quaternion_difference(y_hat, attitude_gt)
                q_error_log = quaternion_log(q_error)
                loss = torch.norm(q_error_log, dim=1, keepdim=False)
                loss = torch.mean(loss, dim=0, keepdim=False)
            
            batch_loss += loss / self.args.unroll_length
            compounding_error.append(loss.detach().cpu().numpy())

            if i < self.args.unroll_length - 1:
                u_gt = y[:, i, -4:]

                if self.args.predictor_type == "velocity":
                    linear_velocity_pred = y_hat[:, :3]
                    angular_velocity_pred = y_hat[:, 3:]
                    attitude_gt = y[:, i, 3:7]
                    
                    if self.args.augment_input == "va":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, va_gt, u_gt), dim=1)
                    elif self.args.augment_input == "w":
                        wind_gt = y[:, i, 10:12]
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, wind_gt, u_gt), dim=1)
                    elif self.args.augment_input == "P":
                        diff_pressure_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, diff_pressure_gt, u_gt), dim=1)
                    elif self.args.augment_input == "vawP":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        wind_gt = y[:, i, 11:13]
                        diff_pressure_gt = y[:, i, 13].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, va_gt, wind_gt, diff_pressure_gt, u_gt), dim=1)
                    else:
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)

                elif self.args.predictor_type == "attitude":
                    linear_velocity_gt = y[:, i, :3]
                    angular_velocity_gt = y[:, i, 7:10]
                    
                    if self.args.augment_input == "va":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, va_gt, u_gt), dim=1)
                    elif self.args.augment_input == "w":
                        wind_gt = y[:, i, 10:12]
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, wind_gt, u_gt), dim=1)
                    elif self.args.augment_input == "P":
                        diff_pressure_gt = y[:, i, 10].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, diff_pressure_gt, u_gt), dim=1)
                    elif self.args.augment_input == "vawP":
                        va_gt = y[:, i, 10].unsqueeze(1)
                        wind_gt = y[:, i, 11:13]
                        diff_pressure_gt = y[:, i, 13].unsqueeze(1)
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, va_gt, wind_gt, diff_pressure_gt, u_gt), dim=1)
                    else:
                        x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)
                
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)

        # Take the average of the trajectory error over the unroll length. trajectory_error_avg_unroll is a list of batch loss for each unroll length
        trajectory_error_avg_unroll = np.mean(trajectory_error_avg_unroll, axis=0)

        self.plot_error.append(trajectory_error_avg_unroll)

        return batch_loss, compounding_error
                
    def quaternion_product(self, delta_q, q):
        """
        Multiply delta quaternion to the previous quaternion.
        """
        
        q = q.unsqueeze(-1)
        delta_q = delta_q.unsqueeze(-1)
        
        # Compute the quaternion product
        q_hat = torch.cat((delta_q[:, 0] * q[:, 0] - delta_q[:, 1] * q[:, 1] - delta_q[:, 2] * q[:, 2] - delta_q[:, 3] * q[:, 3],
                           delta_q[:, 0] * q[:, 1] + delta_q[:, 1] * q[:, 0] + delta_q[:, 2] * q[:, 3] - delta_q[:, 3] * q[:, 2],
                           delta_q[:, 0] * q[:, 2] - delta_q[:, 1] * q[:, 3] + delta_q[:, 2] * q[:, 0] + delta_q[:, 3] * q[:, 1],
                           delta_q[:, 0] * q[:, 3] + delta_q[:, 1] * q[:, 2] - delta_q[:, 2] * q[:, 1] + delta_q[:, 3] * q[:, 0]), dim=-1)
        
        return q_hat.squeeze(-1)
    
    def training_step(self, train_batch, batch_idx):
        _, batch_loss = self.unroll_step(train_batch)

        self.log("train_batch_loss", batch_loss, on_step=True, prog_bar=True, logger=True)

        return batch_loss

    def validation_step(self, valid_batch, batch_idx, dataloader_idx=0):
        
        _, batch_loss = self.unroll_step(valid_batch)
        self.validation_step_outputs.append(batch_loss)
        self.log("valid_batch_loss", batch_loss, on_step=True, prog_bar=True, logger=True)

        return batch_loss
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        
        batch_loss, compounding_error = self.eval_trajectory(test_batch)
        self.compounding_error.append(compounding_error)
        if self.args.predictor_type == "velocity":
            self.log("Velocity Error", batch_loss, on_step=True, prog_bar=True, logger=True)
        elif self.args.predictor_type == "attitude":
            self.log("Quaternion Error", batch_loss, on_step=True, prog_bar=True, logger=True)

        return batch_loss
    
    def plot_compounding_error(self):
        
        # plot the compounding error OVER UNROLL LENGTH
        mean_copounding_error_per_sample = np.mean(self.compounding_error, axis=0)
        var_copounding_error_per_sample = np.var(self.compounding_error, axis=0)
        
        x = np.arange(1, self.args.unroll_length + 1)
        fig, ax = plt.subplots(dpi=100)
        ax.plot(x, mean_copounding_error_per_sample, label="Compounding Error", color='skyblue')
        ax.fill_between(x, mean_copounding_error_per_sample - var_copounding_error_per_sample, 
                        mean_copounding_error_per_sample + var_copounding_error_per_sample, 
                        alpha=0.5, color='skyblue')
        ax.set_xlabel("Unroll Length")
        ax.set_ylabel("MSE")
        ax.set_title("Compounding Error Over Unroll Length")
        # Legend bottom right
        
        plt.tight_layout(pad=1.5)
        plt.savefig(self.experiment_path + "plots/compounding_error.png")
        plt.close(fig)

    def plot_trajectory_error(self):
            
            # plot the error for each sample in the trajectory. All the sample error is stored in self.plot_error as a list of batch loss
            # Iterate the list and plot the error for each sample

            # Compute the total number of samples. It is the sum of the batch size of all the trajectories 
            total_samples = sum([len(error) for error in self.plot_error])

            mse_per_sample = np.concatenate(self.plot_error, axis=0)

            # Plot the error for each sample. MSE vs Sample
            fig, ax = plt.subplots(dpi=100)
            ax.plot(np.arange(1, total_samples + 1), mse_per_sample, label="MSE", color='skyblue', linestyle=line_styles[0], linewidth=1.5)
            ax.set_xlabel("Sample")
            ax.set_ylabel("MSE")
            ax.set_title("MSE for each sample in the trajectory")
            
            # Draw horizontal line at y=1.0
            ax.axhline(y=1.0, color='r', linestyle='--', label='Threshold')

            # Legend bottom right
            plt.tight_layout(pad=1.5)
            plt.grid(alpha=0.3)
            plt.savefig(self.experiment_path + "plots/trajectory_error.png")
            plt.close(fig)

            
            
            
    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        self.verbose = False
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):

        # outputs is a list of tensors that has the loss from each validation step
        avg_loss = torch.stack(self.validation_step_outputs).mean()

        # If validation loss is better than the best validation loss, display the best validation loss
        if avg_loss < self.best_valid_loss:
            self.best_valid_loss = avg_loss
            self.verbose = True
            self.log("best_valid_loss", self.best_valid_loss, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_start(self):
        pass

    def on_test_epoch_end(self):
        
        # get average compounding error over all trajectories and plot the compounding error
        # compounding_error = np.mean(self.compounding_error, axis=0)
        self.plot_compounding_error()
        self.plot_trajectory_error()

        # self.compounding_error.clear()  # free memory
        torch.cuda.empty_cache()
        