import warnings
import torch
import pytorch_lightning
import numpy as np
from .utils import quaternion_difference, quaternion_log

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
    "discrete": ["q_w", "q_x", "q_y", "q_z"],
    "label": ["q_w", "q_x", "q_y", "q_z"],
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

    def forward(self, x, init_memory):

        y_hat = self.model(x, init_memory) # B x 10

        # Multiply predicted delta quaternion to the previous quaternion
        # y_hat is delta quaternion 
        if self.args.delta == True:

            if self.args.predictor_type == "linear_velocity":
                
                if 'v' in self.args.input_type:

                    y_hat = y_hat.clone() + x[:, -1, :3].clone()

            if self.args.predictor_type == "angular_velocity":

                if self.args.input_type == "w":
                    y_hat = y_hat.clone() + x[:, -1, :3].clone()

                if self.args.input_type == "vw":
                    y_hat = y_hat.clone() + x[:, -1, 3:6].clone()

                if self.args.input_type == "wa":
                    y_hat = y_hat.clone() + x[:, -1, 3:6].clone()

                if self.args.input_type == "vwa":
                    y_hat = y_hat.clone() + x[:, -1, 3:6].clone()

            if self.args.predictor_type == "attitude":

                if self.args.input_type == "a":
                    y_hat = self.quaternion_product(y_hat, x[:, -1, :4].clone())

                if self.args.input_type == "va":
                    y_hat = self.quaternion_product(y_hat, x[:, -1, 3:7].clone())
                    
                if self.args.input_type == "wa":
                    y_hat = self.quaternion_product(y_hat, x[:, -1, 3:7].clone())

                if self.args.input_type == "vwa":
                    y_hat = self.quaternion_product(y_hat, x[:, -1, 6:10].clone())

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

            # # Normalize the quaternion
            # if self.args.predictor_type == "attitude":
            #     y_hat = y_hat.clone() / torch.norm(y_hat.clone(), dim=1, keepdim=True)
            
            # loss = self.loss_fn(y_hat, y[:, i, :-4])
            # batch_loss += loss / self.args.unroll_length

            if self.args.predictor_type == "linear_velocity":
                loss = self.loss_fn(y_hat, y[:, i, :3])
                batch_loss += loss / self.args.unroll_length
            
            elif self.args.predictor_type == "angular_velocity":
                loss = self.loss_fn(y_hat, y[:, i, 3:6])
                batch_loss += loss / self.args.unroll_length
            
            else:
                y_hat = y_hat.clone() / torch.norm(y_hat.clone(), dim=1, keepdim=True)
                loss = self.loss_fn(y_hat, y[:, i, 6:10])
                batch_loss += loss / self.args.unroll_length

            if i < self.args.unroll_length - 1:
                
                u_gt = y[:, i, -4:]
                v_gt = y[:, i, :3]
                w_gt = y[:, i, 3:6]
                a_gt = y[:, i, 6:10]


                if self.args.predictor_type == "linear_velocity":
                    
                    if self.args.input_type == "v":
                        x_unroll_curr = torch.cat((y_hat, u_gt), dim=1)

                    elif self.args.input_type == "vw":
                        x_unroll_curr = torch.cat((y_hat, w_gt, u_gt), dim=1)

                    elif self.args.input_type == "va":
                        x_unroll_curr = torch.cat((y_hat, a_gt, u_gt), dim=1)

                    else: 
                        x_unroll_curr = torch.cat((y_hat, w_gt, a_gt, u_gt), dim=1)

                if self.args.predictor_type == "angular_velocity":
                    if self.args.input_type == "w":
                        x_unroll_curr = torch.cat((y_hat, u_gt), dim=1)

                    elif self.args.input_type == "vw":
                        x_unroll_curr = torch.cat((v_gt, y_hat, u_gt), dim=1)

                    elif self.args.input_type == "wa":
                        x_unroll_curr = torch.cat((y_hat, a_gt, u_gt), dim=1)

                    else: 
                        x_unroll_curr = torch.cat((v_gt, y_hat, a_gt, u_gt), dim=1)

                if self.args.predictor_type == "attitude":
                    if self.args.input_type == "a":
                        x_unroll_curr = torch.cat((y_hat, u_gt), dim=1)

                    elif self.args.input_type == "va":
                        x_unroll_curr = torch.cat((v_gt, y_hat, u_gt), dim=1)

                    elif self.args.input_type == "wa":
                        x_unroll_curr = torch.cat((w_gt, y_hat, u_gt), dim=1)

                    else: 
                        x_unroll_curr = torch.cat((v_gt, w_gt, y_hat, u_gt), dim=1)
              
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)
                
            preds.append(y_hat)
            
        return preds, batch_loss
    
    def eval_trajectory(self, test_batch):

        x, y = test_batch
        x = x.float()
        y = y.float()

        x_curr = x 
        batch_loss = 0.0

        for i in range(self.args.unroll_length):

            y_hat = self.forward(x_curr, init_memory=True if i == 0 else False)


            
            if self.args.predictor_type == "linear_velocity":
                loss = self.loss_fn(y_hat, y[:, i, :3])
                batch_loss += loss / self.args.unroll_length
            
            elif self.args.predictor_type == "angular_velocity":
                loss = self.loss_fn(y_hat, y[:, i, 3:6])
                batch_loss += loss / self.args.unroll_length

            else: 
                y_hat = y_hat.clone() / torch.norm(y_hat.clone(), dim=1, keepdim=True)

                attitude_gt = y[:, i, 6:10]

                q_error = quaternion_difference(y_hat, attitude_gt)
                q_error_log = quaternion_log(q_error)

                loss = torch.norm(q_error_log, dim=1, keepdim=False)
                loss = torch.mean(loss, dim=0, keepdim=False)

                batch_loss += loss / self.args.unroll_length

            
            if i < self.args.unroll_length - 1:
                
                u_gt = y[:, i, -4:]
                v_gt = y[:, i, :3]
                w_gt = y[:, i, 3:6]
                a_gt = y[:, i, 6:10]


                if self.args.predictor_type == "linear_velocity":
                    if self.args.input_type == "v":
                        x_unroll_curr = torch.cat((y_hat, u_gt), dim=1)

                    elif self.args.input_type == "vw":
                        x_unroll_curr = torch.cat((y_hat, w_gt, u_gt), dim=1)

                    elif self.args.input_type == "va":
                        x_unroll_curr = torch.cat((y_hat, a_gt, u_gt), dim=1)

                    else: 
                        x_unroll_curr = torch.cat((y_hat, w_gt, a_gt, u_gt), dim=1)

                if self.args.predictor_type == "angular_velocity":
                    if self.args.input_type == "w":
                        x_unroll_curr = torch.cat((y_hat, u_gt), dim=1)

                    elif self.args.input_type == "vw":
                        x_unroll_curr = torch.cat((v_gt, y_hat, u_gt), dim=1)

                    elif self.args.input_type == "wa":
                        x_unroll_curr = torch.cat((y_hat, a_gt, u_gt), dim=1)

                    else: 
                        x_unroll_curr = torch.cat((v_gt, y_hat, a_gt, u_gt), dim=1)

                if self.args.predictor_type == "attitude":
                    if self.args.input_type == "a":
                        x_unroll_curr = torch.cat((y_hat, u_gt), dim=1)

                    elif self.args.input_type == "va":
                        x_unroll_curr = torch.cat((v_gt, y_hat, u_gt), dim=1)

                    elif self.args.input_type == "wa":
                        x_unroll_curr = torch.cat((w_gt, y_hat, u_gt), dim=1)

                    else: 
                        x_unroll_curr = torch.cat((v_gt, w_gt, y_hat, u_gt), dim=1)
              
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)

        return batch_loss

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

    def validation_step(self, valid_batch, batch_idx, dataloader_idx=0):
        
        preds, batch_loss = self.unroll_step(valid_batch)
        self.log("valid_batch_loss", batch_loss, on_step=True, prog_bar=True, logger=True)

        y_hat = preds[0]
        self.val_predictions.append(y_hat.detach().cpu().numpy())
        
        return batch_loss
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        
        batch_loss = self.eval_trajectory(test_batch)
        self.log("Prediction Error", batch_loss, on_step=True, prog_bar=True, logger=True)


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

    def plot_predictions(self, val_predictions, val_full_gt_np=None):
        
        if val_full_gt_np is not None:
            Y_gt = val_full_gt_np
        else: 
            Y_gt = self.val_gt

        # Plot predictions and ground truth
        for i in range(4):
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