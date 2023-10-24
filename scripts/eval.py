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

import sys
import glob
import time 
import os
from tqdm import tqdm
import h5py

def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['X'][:]
        Y = hf['Y'][:]
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

    # set input and output features based on attitude type from args
    if args.attitude == "quaternion":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'e0', 'e1', 'e2', 'e3',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'e0', 'e1', 'e2', 'e3', 
                           'p', 'q', 'r']
    elif args.attitude == "rotation":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'r11', 'r12', 'r13', 
                          'r21', 'r22', 'r23',
                          'r31', 'r32', 'r33',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'r11', 'r21', 'r31', 
                           'r12', 'r22', 'r32',
                           'r13', 'r23', 'r33',
                           'p', 'q', 'r']

        # OUTPUT_FEATURES = ['r11', 'r21', 'r31', 
        #                    'r12', 'r22', 'r32',
        #                    'r13', 'r23', 'r33']
        
    elif args.attitude == "euler":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'phi', 'theta', 'psi',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'phi', 'theta', 'psi',
                           'p', 'q', 'r']
 
    # create the dataset
    X, Y = load_data(data_path + "test/", 'test.h5')

    # convert X and Y to tensors
    X = torch.from_numpy(X).float().to(args.device)
    Y = torch.from_numpy(Y).float().to(args.device)
   
    print(X.shape, Y.shape)
 
    print('Loading model ...')

    # Initialize the model
    
    if args.model_type == "lstm":

        model = LSTM(input_size=len(INPUT_FEATURES), 
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers, 
                    output_size=len(OUTPUT_FEATURES),
                    history_length=args.history_length, 
                    dropout=args.dropout)
    elif args.model_type == "cnn":
        model = CNNModel(input_size=len(INPUT_FEATURES), 
                    num_filters=args.num_filters,
                    kernel_size=args.kernel_size, 
                    output_size=len(OUTPUT_FEATURES),
                    history_length=args.history_length,
                    num_layers=args.num_layers,
                    residual=args.residual,
                    dropout=args.dropout)
    elif args.model_type == "mlp":
        input_size = len(INPUT_FEATURES)
        if args.history_length > 0:
            input_size = input_size * args.history_length
        model = MLP(input_size=input_size, 
                    output_size=len(OUTPUT_FEATURES),
                    num_layers=args.mlp_layers, 
                    dropout=args.dropout)
    elif args.model_type == "tcn":
        model = TCN(num_inputs=len(INPUT_FEATURES), 
                    num_channels=args.num_channels,
                    kernel_size=args.kernel_size, 
                    dropout=args.dropout, 
                    num_outputs=len(INPUT_FEATURES)-4)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)


    input_shape = (1, args.history_length, len(INPUT_FEATURES))
    output_shape = (1, len(OUTPUT_FEATURES))

    input_tensor = X[0:args.history_length].reshape(input_shape)


    Y_plot = np.zeros((X.shape[0] - args.history_length,     Y.shape[1] - 4))
    Y_hat_plot = np.zeros((X.shape[0] - args.history_length, Y.shape[1] - 4))

    print(Y_plot.shape, Y_hat_plot.shape)
    with torch.no_grad():
       for i in range(args.history_length, X.shape[0]):
        
            y_hat = model(input_tensor.flatten()).view(output_shape)           
            x_curr = torch.cat((y_hat, Y[i, -4:].unsqueeze(dim=0)), dim=1) #.clone()

            # print(y_hat[0, 0:3], Y[i, 0:3])
            input_tensor = torch.cat((input_tensor[:, 1:, :], x_curr.view(1, 1, len(INPUT_FEATURES))), dim=1)

            if i < X.shape[0] :
                Y_plot[i - args.history_length, :] = Y[i, :-4].cpu().numpy()
                Y_hat_plot[i - args.history_length, :] = y_hat.cpu().numpy()

                # # PRINT mse loss
                mse_loss = nn.MSELoss()
                loss = mse_loss(y_hat, Y[i, :-4].view(output_shape))
                print("MSE Loss:", loss.item())



    # Plotting on pdf file
    Y_plot = Y_plot[::50, :]
    Y_hat_plot = Y_hat_plot[::50, :]

    with PdfPages(experiment_path + "plots/test.pdf") as pdf:
        for i in range(len(OUTPUT_FEATURES)):
            fig = plt.figure()
            plt.plot(Y_plot[:, i], label="True")
            plt.plot(Y_hat_plot[:, i], label="Predicted")
            plt.xlabel("Time (s)")
            plt.ylabel(OUTPUT_FEATURES[i])
            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)

    
    


    
    
