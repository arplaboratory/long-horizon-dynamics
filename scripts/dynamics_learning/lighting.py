import warnings
import torch
import pytorch_lightning
import numpy as np

from .loss import MSE
from .registry import DynamicsModel

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
        self.model = DynamicsModel(args, input_size, output_size)
            
        self.loss_fn = MSE()
        
        self.best_valid_loss = 1e8
        self.verbose = False

        # Save validation predictions and ground truth
        self.val_predictions = []
        self.val_gt = valid_data
        self.val_full_gt = []

        self.test_predictions = []

    def forward(self, x):
        
        x = self.model(x)
        
        return x
    
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
        x, y = train_batch
        x = x.float()
        y = y.float()
        
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, valid_batch, batch_idx, dataloader_idx=0):
        x, y = valid_batch
        x = x.float()
        y = y.float()

        y_hat = self.forward(x)

        loss = self.loss_fn(y_hat, y)
            
        self.log(f'val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        if self.args.delta == True:
            

            linear_velocity_pred = x[:, -1, :3] + y_hat[:, :3] 
            attitude_pred = y_hat[:, 3:7] * x[:, -1, 3:7]
            angular_velocity_pred = x[:, -1, 7:10] + y_hat[:, 7:10]

            linear_velocity_gt = x[:, -1, :3] + y[:, :3]
            attitude_gt = y[:, 3:7] * x[:, -1, 3:7]

            angular_velocity_gt = x[:, -1, 7:10] + y[:, 7:10]

            # Update the output
            y_hat_full = torch.cat((linear_velocity_pred, attitude_pred, angular_velocity_pred), dim=1)
            y_full = torch.cat((linear_velocity_gt, attitude_gt, angular_velocity_gt), dim=1)
            self.val_full_gt.append(y_full.detach().cpu().numpy())

        self.val_predictions.append(y_hat_full.detach().cpu().numpy())
        
        
        return loss
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        
        # recursive predictions
        if batch_idx == 0:
            x, y = test_batch

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

    def plot_predictions(self, val_predictions, val_full_gt_np=None):
        
        if val_full_gt_np is not None:
            Y_gt = val_full_gt_np
        else: 
            Y_gt = self.val_gt

        # Plot predictions and ground truth
        for i in range(Y_gt.shape[1]):
            fig = plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(val_predictions[:, i], label="Ground Truth", color=colors[1], linewidth=4.5)
            plt.plot(Y_gt[:, i], label="Predicted", color=colors[2], linewidth=4.5,  linestyle=line_styles[1])
            
            plt.grid(True)  # Add gridlines
            plt.tight_layout(pad=1.5)
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel(OUTPUT_FEATURES["label"][i])
            plt.savefig(self.experiment_path + "plots/testset/testset_" + OUTPUT_FEATURES["discrete"][i] + ".png")
            plt.close()

        # release memory
        self.val_predictions = []
        torch.cuda.empty_cache()