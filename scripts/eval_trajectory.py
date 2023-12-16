import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Slider
import seaborn as sns

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

from utils import check_folder_paths, plot_data
from config import parse_args, load_args
from data import DynamicsDataset
from models.lstm import LSTM
from models.cnn import CNNModel
from models.mlp import MLP
from models.tcn import TCN  
from models.tcn_ensemble import TCNEnsemble

import sys
import glob
import time 
import os
from tqdm import tqdm
import h5py


OUTPUT_FEATURES = {
    "euler": ["u", "v", "w", "phi", "theta", "psi", "p", "q", "r"],
    "quaternion": ["u", "v", "w", "q0", "q1", "q2", "q3", "p", "q", "r"],
    "rotation": ["u", "v", "w", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "p", "q", "r"],
    # "test": ["u (m/s)", "v (m/s)", "w (m/s)", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "p (rad/s)", "q (rad/s)", "r (rad/s)"],
    "test": ["u (m/s)", "v (m/s)", "w (m/s)", "q0", "q1", "q2", "q3", "p (rad/s)", "q (rad/s)", "r (rad/s)"],
}


def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['inputs'][:]
        Y = hf['outputs'][:]
    return X, Y

if __name__ == "__main__":

    set_experiment = '/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/experiments/20231216-161226_1/'
    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    if set_experiment is None:
        experiment_path = resources_path + "experiments/" + args.experiment_name + "/"
    else:
        experiment_path = set_experiment 
    model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)

    args = load_args(experiment_path + "args.txt")
    print(experiment_path)
    print("Testing Dynamics model:", model_path)
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
 
    # create the dataset
    X, Y = load_data(data_path + "test/", 'test_trajectory.h5')

    # convert X and Y to tensors
    X = torch.from_numpy(X).float().to(args.device)
    Y = torch.from_numpy(Y).float().to(args.device)

    if args.model_type == "mlp":
        X = X.flatten(1)
    
   
    print(X.shape, Y.shape)
 
    print('Loading model ...')

    # Initialize the model
    
    if args.model_type == "lstm":

        model = LSTM(num_classes=len(OUTPUT_FEATURES[args.attitude]),
                    input_size=len(OUTPUT_FEATURES[args.attitude])+4,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    seq_length=args.history_length)
    elif args.model_type == "cnn":
        model = CNNModel(input_size=len(OUTPUT_FEATURES[args.attitude])+4,
                        num_filters=args.num_filters,
                        kernel_size=args.kernel_size,
                        dropout=args.dropout,
                        num_outputs=len(OUTPUT_FEATURES[args.attitude]))
    elif args.model_type == "mlp":
        model = MLP(input_size=(len(OUTPUT_FEATURES[args.attitude])+4)*args.history_length,
                    output_size=len(OUTPUT_FEATURES[args.attitude]),
                    num_layers=args.mlp_layers,
                    dropout=args.dropout)
        
    elif args.model_type == "tcn":
        model = TCN(num_inputs=len(OUTPUT_FEATURES[args.attitude])+4,
                    num_channels=args.num_channels,
                    kernel_size=args.kernel_size,
                    dropout=args.dropout,
                    num_outputs=len(OUTPUT_FEATURES[args.attitude]))     
    elif args.model_type == "tcn_ensemble":
        model = TCNEnsemble(num_inputs=len(OUTPUT_FEATURES[args.attitude])+4,
                            num_channels=args.num_channels,
                            kernel_size=args.kernel_size,
                            dropout=args.dropout,
                            num_outputs=len(OUTPUT_FEATURES[args.attitude]),
                            ensemble_size=args.ensemble_size)

    model.load_state_dict(torch.load(model_path))
    model.to(args.device)

    mean = torch.from_numpy(np.load(data_path + 'train/' + 'mean_train.npy')).float().to(args.device)
    std = torch.from_numpy(np.load(data_path + 'train/' + 'std_train.npy')).float().to(args.device)

    mean_input = torch.from_numpy(np.load(data_path + 'train/' + 'mean_train_input.npy')).float().to(args.device)
    std_input = torch.from_numpy(np.load(data_path + 'train/' + 'std_train_input.npy')).float().to(args.device)

    input_shape = (1, args.history_length, len(OUTPUT_FEATURES[args.attitude])+4)
    output_shape = (1, len(OUTPUT_FEATURES[args.attitude]))

    input_tensor = X[0:args.history_length].reshape(input_shape)  

    Y_plot = np.zeros((X.shape[0] - args.history_length,     len(OUTPUT_FEATURES[args.attitude])))
    Y_hat_plot = np.zeros((X.shape[0] - args.history_length, len(OUTPUT_FEATURES[args.attitude])))

    # print(Y_plot.shape, Y_hat_plot.shape)
    trajectory_loss = []
    with torch.no_grad():
       for i in range(args.history_length -1, X.shape[0]):
            
            # Normalize the input
            if args.normalize_input:
                input_tensor = (input_tensor - mean_input) / std_input
                
            y_hat = model(input_tensor).view(output_shape) 

            # Unnormalize the output
            if args.normalize:
                y_hat = (y_hat * std[0, :-4]) + mean[0, :-4]

            # Add current input to the output 
            if args.delta:
                # Add velocity
                y_hat[:, 0:3] = y_hat[:, 0:3] + input_tensor[:, -1, 0:3]
                # y_hat[:, 0:3] = y_hat[:, 0:3] + X[i, 0:3]
                # Add angular velocity
                y_hat[:, 12:15] = y_hat[:, 12:15] + input_tensor[:, -1, 12:15]
                # y_hat[:, 12:15] = y_hat[:, 12:15] + X[i, 12:15]

            # # Normalize the quaternion
            q0 = y_hat[:, 3]
            q1 = y_hat[:, 4]
            q2 = y_hat[:, 5]
            q3 = y_hat[:, 6]

            q_norm = torch.sqrt(q0**2 + q1**2 + q2**2 + q3**2).view(-1, 1)

            y_hat[:, 3] = q0 / q_norm
            y_hat[:, 4] = q1 / q_norm
            y_hat[:, 5] = q2 / q_norm
            y_hat[:, 6] = q3 / q_norm

            x_curr = torch.cat((y_hat, Y[i, -4:].unsqueeze(dim=0)), dim=1) #.clone()

            input_tensor = torch.cat((input_tensor[:, 1:, :], x_curr.view(1, 1, len(OUTPUT_FEATURES[args.attitude])+4)), dim=1)

            if i < X.shape[0] :

                Y_plot[i - args.history_length, :] = Y[i, :-4].cpu().numpy()
                Y_hat_plot[i - args.history_length, :] = y_hat.cpu().numpy()

                # # PRINT mse loss
                mse_loss = nn.MSELoss()
                loss = mse_loss(y_hat, Y[i, :-4].view(output_shape))
                trajectory_loss.append(loss.item())
            
    mean_loss = np.mean(trajectory_loss)
    print("MSE Loss: ", mean_loss)
    
    # Y_plot = Y_plot[::50, :]
    # Y_hat_plot = Y_hat_plot[::50, :]

    # Y_plot = Y_plot[:10, :]
    # Y_hat_plot = Y_hat_plot[:10, :]

    ### Plot and Save MSE loss as a histogram
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(trajectory_loss, bins=30, color='skyblue', alpha=0.7, edgecolor='black')

    ax.set_xlabel("MSE Loss")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of MSE Loss")

    # Adding gridlines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Customize ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='x', rotation=45)

    # Save the plot
    plt.tight_layout()
    plt.savefig(experiment_path + "plots/trajectory/eval_mse_histogram.png")
    plt.close()

    ### Plot and Save MSE loss over no. of recursions
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(trajectory_loss, color='skyblue', linewidth=2.5, label='MSE Loss')

    # Plot mean loss line
    ax.axhline(y=mean_loss, label="Mean Loss", color='orange', linestyle=line_styles[1], linewidth=2.5)

    # Print mean and variance on the plot
    # ax.text(0.80, 0.9, f"Mean Loss: {mean_loss:.3f}", transform=ax.transAxes)
    # ax.text(0.80, 0.85, f"Variance: {np.var(trajectory_loss):.3f}", transform=ax.transAxes)

    ax.set_xlabel("No. of Recursive Predictions")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MSE Loss Analysis")
    ax.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(experiment_path + "plots/trajectory/eval_mse_loss.png")
    plt.close()

    ### Plot and Save MSE loss as a boxplot 

    plt.figure(figsize=(8, 6))
    sns.boxplot(y=trajectory_loss, color='skyblue')
    plt.xlabel("MSE Loss")
    plt.title("Distribution of MSE Loss")
    plt.savefig(experiment_path + "plots/trajectory/eval_mse_loss_boxplot.png")
    plt.close()

    ### Plot and Save MSE loss as a density plot 
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=trajectory_loss, shade=True, color='skyblue')
    plt.xlabel("MSE Loss")
    plt.title("Density Plot of MSE Loss")
    plt.savefig(experiment_path + "plots/trajectory/eval_mse_loss_density.png")
    plt.close()

    # Generate aesthetic plots and save them individually
    # Generate aesthetic plots and save them individually
    for i in range(10):
        fig = plt.figure(figsize=(8, 6), dpi=400)
        plt.plot(Y_plot[:, i], label="Ground Truth", color=colors[1], linewidth=4.5)
        plt.plot(Y_hat_plot[:, i], label="Predicted", color=colors[2], linewidth=4.5,  linestyle=line_styles[1])
        
        plt.grid(True)  # Add gridlines
        plt.tight_layout(pad=1.5)
        plt.legend()
        plt.xlabel("No. of recursive predictions")
        plt.ylabel(OUTPUT_FEATURES["test"][i])
        plt.savefig(experiment_path + "plots/trajectory/trajectory_" + OUTPUT_FEATURES[args.attitude][i] + ".png")
        plt.close()