import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
}


def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['inputs'][:]
        Y = hf['outputs'][:]
    return X, Y

if __name__ == "__main__":

    set_experiment = '/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/experiments/20231206-170703_1/'
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
    X, Y = load_data(data_path + "test/", 'test.h5')

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

    input_shape = (1, args.history_length, len(OUTPUT_FEATURES[args.attitude])+4)
    output_shape = (1, len(OUTPUT_FEATURES[args.attitude]))

    input_tensor = X[0:args.history_length].reshape(input_shape)  

    Y_plot = np.zeros((X.shape[0] - args.history_length,     len(OUTPUT_FEATURES[args.attitude])))
    Y_hat_plot = np.zeros((X.shape[0] - args.history_length, len(OUTPUT_FEATURES[args.attitude])))

    # print(Y_plot.shape, Y_hat_plot.shape)
    trajectory_loss = []
    with torch.no_grad():
       for i in range(args.history_length -1, X.shape[0]):
        
            y_hat = model(input_tensor).view(output_shape) 

            # Reortho-normalize rotation matrix using SVD
            R_column1 = [y_hat[0][3], y_hat[0][6], y_hat[0][9]]
            R_column2 = [y_hat[0][4], y_hat[0][7], y_hat[0][10]]
            R_column3 = [y_hat[0][5], y_hat[0][8], y_hat[0][11]]

            # SVD 
            U, S, V = torch.svd(torch.tensor([R_column1, R_column2, R_column3]))

            # Reconstruct rotation matrix
            R = torch.mm(U, V.t())

            # Reconstruct output
            y_hat[0][3] = R[0][0]
            y_hat[0][4] = R[0][1]
            y_hat[0][5] = R[0][2]
            y_hat[0][6] = R[1][0]
            y_hat[0][7] = R[1][1]
            y_hat[0][8] = R[1][2]
            y_hat[0][9] = R[2][0]
            y_hat[0][10] = R[2][1]
            y_hat[0][11] = R[2][2]
            
         
            # Add current input to the output 
            y_hat = input_tensor[:, -1, :-4] + y_hat
       

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
    
    Y_plot = Y_plot[::50, :]
    Y_hat_plot = Y_hat_plot[::50, :]


    # Plot and Save MSE loss and mean MSE loss
    fig = plt.figure()
    plt.plot(trajectory_loss)
    # PLot mean loss
    plt.plot(np.ones(len(trajectory_loss))*mean_loss, label="Mean Loss")
    plt.legend()
    # print to precision 3 of mean and varience to plot top left
    plt.text(0, 0.95, f"Mean Loss: {mean_loss:.3f}", transform=plt.gca().transAxes)
    plt.text(0, 0.90, f"Variance: {np.var(trajectory_loss):.3f}", transform=plt.gca().transAxes)

    plt.xlabel("No. of recursive predictions")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss")
    plt.savefig(experiment_path + "plots/eval_mse_loss.pdf")

    with PdfPages(experiment_path + "plots/eval_trajectory.pdf") as pdf:
        for i in range(len(OUTPUT_FEATURES[args.attitude])):
            fig = plt.figure()
            plt.plot(Y_plot[:, i], label="True")
            plt.plot(Y_hat_plot[:, i], label="Predicted")
            plt.xlabel("Time (s)")
            plt.ylabel(OUTPUT_FEATURES[args.attitude][i])
            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)