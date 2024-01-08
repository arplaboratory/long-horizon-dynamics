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
    experiment_path = experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
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
        output_size=4,
        valid_data=Y,
        max_iterations=1,
    )

    # # Load the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(args.device)

    input_shape = (1, args.history_length, X.shape[-1])
    output_shape = (1, 4)

    # Y_plot = np.zeros((X.shape[0] - args.history_length,     6))
    # Y_hat_plot = np.zeros((X.shape[0] - args.history_length, 6))

    mse_loss = MSE()
    sample_loss = []

    copounding_error_per_sample = []
    mean_abs_error_per_sample = []  

    model.eval()
    with torch.no_grad():
        
        for i in tqdm(range(args.history_length, X.shape[0])):

            x = X[i, :, :]
            y = Y[i, :, :]

            x = x.unsqueeze(0)
            x_curr = x 
            batch_loss = 0.0

                        
            abs_error = []

            compounding_error = []

            for j in range(args.unroll_length):
                
                y_hat = model.forward(x_curr, init_memory=True if j == 0 else False)

                # Normalize the quaternion
                y_hat = y_hat / torch.norm(y_hat, dim=1, keepdim=True)
                
                attitude_gt = y[j, 3:7]

                abs_error.append(torch.abs(y_hat - attitude_gt))

                loss = mse_loss(y_hat, attitude_gt)
                batch_loss += loss / args.unroll_length

                compounding_error.append(loss.item())

                if j < args.unroll_length - 1:
                    
                    linear_velocity_gt = y[j, :3].view(1, 3)
                    angular_velocity_gt = y[j, 7:10].view(1, 3)

                    u_gt = y[j, -4:].unsqueeze(0)

                    # Update x_curr
                    x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)
                
                    x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)

            mean_abs_error_per_sample.append(torch.mean(torch.cat(abs_error, dim=0), dim=0).cpu().numpy())
            sample_loss.append(batch_loss.item())
            copounding_error_per_sample.append(compounding_error)

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
    ax.set_ylabel("MSE")
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
    ax.set_ylabel("MSE")
    # Set title with the unroll length
    ax.set_title(f"MSE over {args.unroll_length} Recursive Predictions")

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
