import warnings
import torch
import pytorch_lightning
import numpy as np
from .utils import quaternion_difference, quaternion_log

from .loss import MSE
from .registry import get_model

warnings.filterwarnings("ignore")

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

                    # Update x_curr
                    x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)

                elif self.args.predictor_type == "attitude":
                    linear_velocity_gt = y[:, i, :3]
                    angular_velocity_gt = y[:, i, 7:10]
                    # Update x_curr
                    x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)
                
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

            if self.args.predictor_type == "velocity":
                linear_velocity_gt =  y[:, i, :3]
                angular_velocity_gt = y[:, i, 7:10]
                velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=1)
                abs_error[i+1] = torch.mean(torch.abs(y_hat - velocity_gt), dim=0)
                loss = self.loss_fn(y_hat, velocity_gt)
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
                    x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)

                elif self.args.predictor_type == "attitude":
                    linear_velocity_gt = y[:, i, :3]
                    angular_velocity_gt = y[:, i, 7:10]
                    x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)
                
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)

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
        
        batch_loss, _ = self.eval_trajectory(test_batch)
        self.log("MSE", batch_loss, on_step=True, prog_bar=True, logger=True)

        return batch_loss
            
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

        