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
    elif args.attitude == "euler":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'phi', 'theta', 'psi',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'phi', 'theta', 'psi',
                           'p', 'q', 'r']
 
    # create the dataset
    test_dataset = DynamicsDataset(data_path + "test/", 'test.h5', args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

   

    # print number of datapoints
    print("Shape of input:", test_dataset.X_shape)
    print("Shape of output:", test_dataset.Y_shape)

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
        model = MLP(input_size=len(INPUT_FEATURES), 
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
    
    batch = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing")
    model.eval()

    if args.history_length == 0:
        Y = np.zeros((test_dataset.Y.shape[1], test_dataset.Y.shape[0]))
        Y_hat = np.zeros((test_dataset.Y.shape[1], test_dataset.Y.shape[0]))
    else:
        Y = np.zeros((test_dataset.X.shape[2], test_dataset.Y.shape[0]))
        Y_hat = np.zeros((test_dataset.X.shape[2], test_dataset.Y.shape[0]))

    # Inference speed computation
    with torch.no_grad():
        test_x, test_y = next(iter(test_dataloader))
        test_x = test_x.to(args.device).float()
        test_y = test_y.to(args.device).float()
        frequency_hist = []
        for k in range(11):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            y_pred = model(test_x)
            ender.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            if k:
                frequency_hist.append(1. / (starter.elapsed_time(ender) / 1000))
        print("Inference speed (Hz): ", np.mean(frequency_hist))
    
    # Inference
    # with torch.no_grad():

        # for i, (x, y) in enumerate(batch):
        #     x = x.to(args.device).float()
        #     y = y.to(args.device).float()

        #     y_pred = model(x)

        #     Y[i*args.batch_size:(i+1)*args.batch_size, :] = y.cpu().numpy()
        #     Y_hat[i*args.batch_size:(i+1)*args.batch_size, :] = y_pred.cpu().numpy()
    
    # Get initial state, control action and propagate it through the model
    with torch.no_grad():
        for i, (x, y) in enumerate(batch):
            x = x.to(args.device).float()
            y = y.to(args.device).float()

            for j in range(x.shape[0]):

                if args.history_length == 0:
                    if j == 0:
                        y_pred = model(x[j, :].unsqueeze(0))
                        Y[j, :] = y[j, :].cpu().numpy()
                        Y_hat[j, :] = y_pred.cpu().numpy()
                    else:

                        # copy y_pred from previous time step
                        x_now = y_pred.clone()
                        u_curr = x[j, -4:].unsqueeze(0)
                    

                        y_pred = model(torch.cat((y_pred, u_curr), dim=1))

                        # Save y_pred and y at every time step and plot it
                        Y[j, :] = y[j, :].cpu().numpy()
                        Y_hat[j, :] = y_pred.cpu().numpy()
                else:
                    if j == 0:
                        y_pred = model(x[j, :, :].unsqueeze(0))
                        Y[j, :] = y[j, :].cpu().numpy()
                        Y_hat[j, :] = y_pred.cpu().numpy()
                    else:

                        # copy y_pred from previous time step
                        x_now = y_pred.clone()

                        # update x with the new state x_now 
                        x[j, -1, :x_now.shape[1]] = x_now[0, :]
                        y_pred = model(x[j, :, :].unsqueeze(0))

                        # # Save y_pred and y at every time step and plot it
                        Y[j, :] = y[j, :].cpu().numpy()
                        Y_hat[j, :] = y_pred.cpu().numpy()

           


    # Plotting on pdf file
    
    Y = Y[::50, :]
    Y_hat = Y_hat[::50, :]

    with PdfPages(experiment_path + "plots/test.pdf") as pdf:
        for i in range(len(OUTPUT_FEATURES)):
            fig = plt.figure()
            plt.plot(Y[:, i], label="True")
            plt.plot(Y_hat[:, i], label="Predicted")
            plt.xlabel("Time (s)")
            plt.ylabel(OUTPUT_FEATURES[i])
            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)

    
    


    
    
