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
    "test": ["u (m/s)", "v (m/s)", "w (m/s)", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "p (rad/s)", "q (rad/s)", "r (rad/s)"],
}


def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['inputs'][:]
        Y = hf['outputs'][:]
    return X, Y

if __name__ == "__main__":

    
    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
    model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)

    args = load_args(experiment_path + "args.txt")
    print(experiment_path)
    print("Testing Dynamics model:", model_path)

    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
 
    # create the dataset
    X, Y = load_data(data_path + "test/", 'test_eval.h5')
   
    # convert X and Y to tensors
    X = torch.from_numpy(X).float().to(args.device)
    Y = torch.from_numpy(Y).float().to(args.device)

    Y_plot = np.zeros((X.shape[0] - args.history_length,     Y.shape[2] - 4))
    Y_hat_plot = np.zeros((X.shape[0] - args.history_length, Y.shape[2] - 4))

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
    
    model.eval()

    # Inference
    with torch.no_grad():

        output = model(X)        


        loss = torch.mean((output - Y[:, 0, :-4])**2)
        print("Test Loss: ", loss.item())

        # print variance
        print("Variance: ", torch.var(output - Y[:, 0, :-4]).item())

    
        # Plot predictions and ground truth
        Y_plot = Y[:, 0, :-4].detach().cpu().float().numpy()
        Y_hat_plot = output.detach().cpu().float().numpy()

        # print('--------------------------')
        # print(Y_plot[-100:])
        # print(Y_hat_plot[-100:])

        Y_plot = Y_plot[::50, :]
        Y_hat_plot = Y_hat_plot[::50, :]

        # Generate aesthetic plots and save them individually
        for i in range(15):
            fig = plt.figure(figsize=(16, 16), dpi=400)
            plt.plot(Y_plot[:, i], label="Ground Truth", color=colors[1])
            plt.plot(Y_hat_plot[:, i], label="Predicted", color=colors[2])
            plt.legend()
            plt.xlabel("No. of recursive predictions")
            plt.ylabel(OUTPUT_FEATURES["test"][i])
            plt.savefig(experiment_path + "plots/testset_" + OUTPUT_FEATURES[args.attitude][i] + ".png")
            plt.close()


    
    
   


    
    