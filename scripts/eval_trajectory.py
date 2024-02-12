import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider
import seaborn as sns


from config import parse_args, load_args
from dynamics_learning.loss import MSE
from dynamics_learning.lighting import DynamicsLearning
from copy import deepcopy

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
# colors = plt.cm.tab10(np.linspace(0, 1, 6)) * 0.2  # Multiplying by 0.5 for darker shades

from config import load_args
from dynamics_learning.utils import check_folder_paths

import sys
import glob
import time 
import os
from tqdm import tqdm
import h5py


def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['inputs'][:]
        Y = hf['outputs'][:]
    return X, Y

def quaternion_product(q1, q2):
    
        # Compute the product of two quaternions
        # Input: q1 = [q_w, q_x, q_y, q_z]
        #        q2 = [q_w, q_x, q_y, q_z]
    
        w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:]
        w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:]

        # Compute the product of the two quaternions
        q_prod = torch.cat((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dim=1)
        
        return q_prod

def quaternion_difference(q_pred, q_gt):

    # Compute the difference between two quaternions
    # Input: q_pred = [q_w, q_x, q_y, q_z]
    #        q_gt   = [q_w, q_x, q_y, q_z]

    # Compute the norm of the quaternion
    norm_q_pred = torch.norm(q_pred, dim=1, keepdim=True) 
    norm_q_gt = torch.norm(q_gt, dim=1, keepdim=True) 

    # Normalize the quaternion
    q_pred = q_pred / norm_q_pred
    q_gt = q_gt / norm_q_gt

    # q_pred inverse
    q_pred_inv = torch.cat((q_pred[:, 0:1], -q_pred[:, 1:]), dim=1)

    # Compute the difference between the two quaternions
    q_diff = quaternion_product(q_gt, q_pred_inv)

    return q_diff

def quaternion_log(q):
        
        # Compute the log of a quaternion
        # Input: q = [q_w, q_x, q_y, q_z]
        
        # Compute the norm of the quaternion
        
        # norm_q = torch.norm(q, dim=1, keepdim=True) 
        
        # Get vector part of the quaternion
        q_v = q[:, 1:]
        q_v_norm = torch.norm(q_v, dim=1, keepdim=True)
        
        # Compute the angle of rotation
        theta = 2 * torch.atan2(q_v_norm, q[:, 0:1])
        
        # Compute the log of the quaternion
        q_log = theta * q_v / q_v_norm
        
        return q_log

def quaternion_error(q_pred, q_gt):

    # Compute dot product between two quaternions
    # Input: q_pred = [q_w, q_x, q_y, q_z]
    #        q_gt   = [q_w, q_x, q_y, q_z]

    # Compute the norm of the quaternion
    norm_q_pred = torch.norm(q_pred, dim=1, keepdim=True)
    norm_q_gt = torch.norm(q_gt, dim=1, keepdim=True)

    # Normalize the quaternion
    q_pred = q_pred / norm_q_pred
    q_gt = q_gt / norm_q_gt

    # Compute the dot product between the two quaternions 
    q_dot = torch.sum(q_pred * q_gt, dim=1, keepdim=True)

    # Compute the angle between the two quaternions
    theta = torch.acos(torch.abs(q_dot))

    # min_theta = torch.min(theta, np.pi - theta)

    # COnvert to degrees
    # theta = theta * 180 / np.pi

    return theta

if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # Asser model type
    assert args.model_type in ["mlp", "lstm", "gru", "tcn", "transformer"], "Model type must be one of [mlp, lstm, gru, tcn, transformer]"

    # Assert attitude type
    assert args.attitude in ["euler", "quaternion", "rotation"], "Attitude type must be one of [euler, quaternion, rotation]"

    # Seed
    pytorch_lightning.seed_everything(args.seed)

    # Assert vehicle type
    assert args.vehicle_type in ["fixed_wing", "quadrotor"], "Vehicle type must be one of [fixed_wing, quadrotor]"

    if args.vehicle_type == "fixed_wing":
        vehicle_type = "fixed_wing"
    else:
        vehicle_type = "quadrotor"

    # Set global paths
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/" + vehicle_type + "/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
    model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)

    check_folder_paths([os.path.join(experiment_path, "checkpoints"), os.path.join(experiment_path, "plots"), os.path.join(experiment_path, "plots", "trajectory"), 
                        os.path.join(experiment_path, "plots", "testset")])

    print(experiment_path)
    print("Testing Dynamics model:", model_path)
    args = load_args(experiment_path + "args.txt")
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
 
    # create the dataset
    X, Y = load_data(data_path + "test/", 'test.h5')

    # convert X and Y to tensors
    X = torch.from_numpy(X).float().to(args.device)
    Y = torch.from_numpy(Y).float().to(args.device)

    print(X.shape, Y.shape)
 
    print('Loading model ...')

    # Initialize the model
    
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=X.shape[-1],
        output_size=10,
        valid_data=Y,
        max_iterations=1,
    )

    # # Load the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(args.device)

    input_shape = (1, args.history_length, X.shape[-1])
    output_shape = (1, 10)

    # Y_plot = np.zeros((X.shape[0] - args.history_length,     6))
    # Y_hat_plot = np.zeros((X.shape[0] - args.history_length, 6))

    mse_loss = MSE()
    sample_loss_attitude = []
    sample_loss_velocity = []

    copounding_error_per_sample = []
    mean_abs_error_per_sample = []  

    velocity_error = []
    attitude_error = []

    model.eval()
    with torch.no_grad():
        
        for i in tqdm(range(args.history_length, X.shape[0])):

            x = X[i, :, :]
            y = Y[i, :, :]

            x = x.unsqueeze(0)
            x_curr = x 
            batch_loss_attitude = 0.0
            batch_loss_velocity = 0.0

                        
            abs_error_velocity = []
            abs_error_attitude = []

            compounding_error_velocity = []
            compounding_error_attitude = []

            for j in range(args.unroll_length):
                
                y_hat = model.forward(x_curr, init_memory=True if j == 0 else False)
                
                
                linear_velocity_pred = y_hat[:, :3]
                attitude_pred = y_hat[:, 3:7]
                angular_velocity_pred = y_hat[:, 7:10]
                velocity_pred = torch.cat((linear_velocity_pred, angular_velocity_pred), dim=1)
               
                
                linear_velocity_gt = y[j, :3].view(-1, 3)
                attitude_gt = y[j, 3:7]
                angular_velocity_gt = y[j, 7:10].view(-1, 3)
                velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=1)

                
                q_error = quaternion_difference(attitude_pred, attitude_gt.unsqueeze(0))
                q_error_log = quaternion_log(q_error)

                # q_error = quaternion_error(y_hat, attitude_gt.unsqueeze(0))
                # abs_error.append(torch.norm(q_error_log, dim=1, keepdim=True))
                abs_error_attitude.append(torch.abs(q_error_log))

                
                loss_attitde = torch.norm(q_error_log, dim=1, keepdim=False)[0] #mse_loss(q_error_log)
                batch_loss_attitude += loss_attitde / args.unroll_length
                compounding_error_attitude.append(loss_attitde.cpu().numpy())

                loss_velocity = mse_loss(velocity_pred, velocity_gt)
                batch_loss_velocity += loss_velocity / args.unroll_length
                compounding_error_velocity.append(loss_velocity.cpu().numpy())

                # Normalize the quaternion
                attitude_pred = attitude_pred / torch.norm(attitude_pred, dim=1, keepdim=True)


                if j < args.unroll_length - 1:
                    
                    u_gt = y[j, -4:].unsqueeze(0)

                    # Update x_curr
                    x_unroll_curr = torch.cat((linear_velocity_pred, attitude_pred, angular_velocity_pred, u_gt), dim=1)
                
                    x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)

            # mean_abs_error_per_sample.append(torch.mean(torch.cat(abs_error, dim=0), dim=0).cpu().numpy())
            sample_loss_attitude.append(batch_loss_attitude.item())
            sample_loss_velocity.append(batch_loss_velocity.item())

    
    
    
    print("Mean Attitude Error per sample: ", np.mean(sample_loss_attitude))
    print("Variance Attitude Error per sample: ", np.var(sample_loss_attitude))
    print("Mean Velocity Error per sample: ", np.mean(sample_loss_velocity))
    print("Variance Velocity Error per sample: ", np.var(sample_loss_velocity))

    '''
    #################################################################################################################################################
    copounding_error_per_sample = np.array(copounding_error_per_sample)
    print("Mean Copounding Error per sample: ", np.mean(copounding_error_per_sample, axis=0))
    # Print varience of copounding error per sample
    print("Variance Copounding Error per sample: ", np.var(copounding_error_per_sample, axis=0))
    # Mean and overlay variance of copounding error per sample over number of recursions
    mean_copounding_error_per_sample = np.mean(copounding_error_per_sample, axis=0)
    var_copounding_error_per_sample = np.var(copounding_error_per_sample, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(mean_copounding_error_per_sample, color='skyblue', linewidth=2.5, label='Mean Copounding Error')
    ax.fill_between(np.arange(len(mean_copounding_error_per_sample)), mean_copounding_error_per_sample - var_copounding_error_per_sample, 
                    mean_copounding_error_per_sample + var_copounding_error_per_sample, alpha=0.5, color='skyblue', 
                    label='Variance Copounding Error')

    ax.set_xlabel("No. of Recursive Predictions")
    ax.set_ylabel("Quaternion Error")
    ax.set_title("MSE Analysis over Recursive Predictions")
    ax.legend()

    # Save the plot
    plt.tight_layout(pad=1.5)
    plt.savefig(experiment_path + "plots/trajectory/eval_mse_loss.png")
    plt.close()

    #################################################################################################################################################

    # Print the mean_abs_error_per_sample over the entire test set
    mean_abs_error_per_sample = np.array(mean_abs_error_per_sample)
    print("Mean Absolute Error per sample: ", np.mean(mean_abs_error_per_sample, axis=0))
    
    # Plot sample loss as bar plot over every sample 
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.bar(np.arange(len(sample_loss)), sample_loss, color='skyblue', alpha=0.7, edgecolor='black')

    ax.set_xlabel("Sample")
    ax.set_ylabel("Quaternion Error")
    # Set title with the unroll length
    # ax.set_title(f"Mean Quaternion Error (rad) over {args.unroll_length} Recursive Predictions")

    # Adding gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Customize ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='x', rotation=45)

    # Save the plot
    plt.tight_layout(pad=1.5)
    plt.savefig(experiment_path + "plots/trajectory/eval_mse_sample_loss.png")
    plt.close()

    #################################################################################################################################################
'''