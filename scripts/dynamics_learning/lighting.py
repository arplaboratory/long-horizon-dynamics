import warnings
import torch
import pytorch_lightning
import numpy as np

from .loss import MSE
from .registry import get_model

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (19.20, 10.80)
# font = {"family" : "sans",
#         "weight" : "normal",
#         "size"   : 28}
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # Choose your serif font here
    "font.size": 28,
    # "figure.figsize": (19.20, 10.80),
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# matplotlib.rc("font", **font)
# matplotlib.rcParams["pdf.fonttype"] = 42
# matplotlib.rcParams["ps.fonttype"] = 42
colors = ["#7d7376","#365282","#e84c53","#edb120"]
markers = ['o', 's', '^', 'D', 'v', 'p']
line_styles = ['-', '--', '-.', ':', '-', '--']


OUTPUT_FEATURES = {
    "discrete": ["v_x", "v_y", "v_z", "q_w", "q_x", "q_y", "q_z", "w_x", "w_y", "w_z", "u_0", "u_1", "u_2", "u_3"],
    "label": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "q_w", "q_x", "q_y", "q_z", "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"],
}

class DynamicsLearning(pytorch_lightning.LightningModule):
    def __init__(self, args, resources_path, experiment_path, 
                 input_size, output_size, valid_data, max_iterations):
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

        # Save validation predictions and ground truth
        self.val_predictions = []
        self.val_gt = valid_data
        self.val_full_gt = []

        self.test_predictions = []
        self.mean_abs_error_per_sample = []
        self.copounding_error_per_sample = []

    def forward(self, x, init_memory):

        y_hat = self.model(x, init_memory) 

        # Add predicted delta linear velocity and angular velocity

        if self.args.delta == True:
            y_hat[:, :3] = y_hat[:, :3] + x[:, -1, :3]
            y_hat[:, 3:] = y_hat[:, 3:] + x[:, -1, 7:10]
        
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

    def training_step(self, train_batch, batch_idx):
        _, batch_loss = self.unroll_step(train_batch)

        self.log("train_batch_loss", batch_loss, on_step=True, prog_bar=True, logger=True)

        return batch_loss
    
    def unroll_step(self, batch):

        x, y = batch
        x = x.float()
        y = y.float()

        x_curr = x 
        preds = []

        batch_loss = 0.0
        for i in range(self.args.unroll_length):
            y_hat = self.forward(x_curr, init_memory=True if i == 0 else False)
            # y_gt = y[:, i, :self.output_size]

            linear_velocity_gt =  y[:, i, :3]
            angular_velocity_gt = y[:, i, 7:10]

            velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=1)

            loss = self.loss_fn(y_hat, velocity_gt)
            batch_loss += loss / self.args.unroll_length

            if i < self.args.unroll_length - 1:
                
                linear_velocity_pred = y_hat[:, :3]
                angular_velocity_pred = y_hat[:, 3:]

                u_gt = y[:, i, -4:]
                attitude_gt = y[:, i, 3:7]

                # Update x_curr
                x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)
              
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
        abs_error = {}

        for i in range(self.args.unroll_length):
            y_hat = self.forward(x_curr, init_memory=True if i == 0 else False)
      
            linear_velocity_gt =  y[:, i, :3]
            angular_velocity_gt = y[:, i, 7:10]

            velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=1)

            abs_error[i+1] = torch.mean(torch.abs(y_hat - velocity_gt), dim=0)

            loss = self.loss_fn(y_hat, velocity_gt, mean=False)
            batch_loss += loss.mean(0) / self.args.unroll_length

            # Mean absolute error


            compounding_error.append(loss.detach().cpu().numpy())

            if i < self.args.unroll_length - 1:
                
                linear_velocity_pred = y_hat[:, :3]
                angular_velocity_pred = y_hat[:, 3:]

                u_gt = y[:, i, -4:]
                attitude_gt = y[:, i, 3:7]

                # Update x_curr
                x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)
              
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)
        
        
        self.mean_abs_error_per_sample.append(abs_error)
                
        return batch_loss, compounding_error  

    def validation_step(self, valid_batch, batch_idx, dataloader_idx=0):
        
        _, batch_loss = self.unroll_step(valid_batch)
        self.log("valid_batch_loss", batch_loss, on_step=True, prog_bar=True, logger=True)
        
        return batch_loss
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        
        batch_loss, compounding_error = self.eval_trajectory(test_batch)
        self.log("MSE", batch_loss, on_step=True, prog_bar=True, logger=True)

        # compounding_error is of shape (unroll_length, batch_size)
        # At the end of evvery step concatenate the compounding error on the batch axis

        
        self.copounding_error_per_sample.append(np.array(compounding_error))

        return batch_loss
            
    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        self.verbose = False
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        pass

    def validation_epoch_end(self, outputs):

        # outputs is a list of tensors that has the loss from each validation step
        avg_loss = torch.stack(outputs).mean()

        # If validation loss is better than the best validation loss, display the best validation loss
        if avg_loss < self.best_valid_loss:
            self.best_valid_loss = avg_loss
            self.verbose = True
            self.log("best_valid_loss", self.best_valid_loss, on_epoch=True, prog_bar=True, logger=True)

        # Plotting based on the validation frequency
        # if self.current_epoch % self.args.plot_freq == 0:
        #     # Plot validation predictions
        #     val_predictions_np = np.concatenate(self.val_predictions, axis=0)

        #     if self.args.delta == True:
        #         val_full_gt_np = np.concatenate(self.val_full_gt, axis=0)
        #         self.plot_predictions(val_predictions_np, val_full_gt_np)
        #     else:
        #         # Plot predictions and ground truth
        #         self.plot_predictions(val_predictions_np)
            
    def on_test_epoch_end(self) -> None:

        # self.compounding_error_per_sample concatenates the compounding error on the batch axis
        recursive_predictions_per_sample = np.concatenate(self.copounding_error_per_sample, axis=1)

        # recursive predictions per sample is of shape (unroll_length, batch_size). Flip the axis to get (batch_size, unroll_length)
        recursive_predictions_per_sample = np.transpose(recursive_predictions_per_sample, (1, 0))

        # Save recursive_predictions_per_sample as numpy array
        np.save(self.experiment_path + "plotting_data/" + self.args.model_type + ".npy", recursive_predictions_per_sample)

        # Get mean and variance of compound error per sample
        mean_copounding_error_per_sample = np.mean(recursive_predictions_per_sample, axis=0)
        variance_copounding_error_per_sample = np.var(recursive_predictions_per_sample, axis=0)

        # plot the evaluations
        self.plot_evaluations(mean_copounding_error_per_sample, variance_copounding_error_per_sample)


        
        
    def plot_evaluations(self, mean_copounding_error_per_sample, variance_copounding_error_per_sample):
        

        # Plot the data
        fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
        ax.plot(mean_copounding_error_per_sample, color='skyblue', linewidth=2.5, label='Mean Copounding Error')
        ax.fill_between(np.arange(len(mean_copounding_error_per_sample)), mean_copounding_error_per_sample - variance_copounding_error_per_sample, 
                        mean_copounding_error_per_sample + variance_copounding_error_per_sample, alpha=0.5, color='skyblue', 
                        label='Variance Copounding Error')
    
        # Set axis limits
        # ax.set_xlim([0, 20])
        # ax.set_ylim([0, 10])

        ax.set_xlabel("No. of Recursive Predictions")
        ax.set_ylabel("MSE")
        ax.set_title("MSE Analysis over Recursive Predictions")
        ax.legend()

        # Save the plot
        plt.tight_layout(pad=1.5)
        plt.savefig(self.experiment_path + "plots/trajectory/mse_analysis.pdf")
        plt.close()


   