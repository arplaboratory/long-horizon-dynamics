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
from config import parse_args
from data import DynamicsDataset
from models.lstm import LSTM
from models.cnn import CNNModel
from models.mlp import MLP

import sys
import glob
import time 
import os
from tqdm import tqdm


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
    model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)

    print("Testing Dynamics model:", model_path)
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id))

    INPUT_FEATURES = ['u', 'v', 'w',
                      'e0', 'e1', 'e2', 'e3',
                      'p', 'q', 'r',
                      'delta_e', 'delta_a', 'delta_r', 'delta_t']
    OUTPUT_FEATURES = ['u', 'v', 'w',
                       'p', 'q', 'r',]
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id) + "\n")

    # create the dataset
    test_dataset = DynamicsDataset(data_path + "test/", args.batch_size, INPUT_FEATURES, OUTPUT_FEATURES, 
                                    history_length=args.history_length, normalize=args.normalize, 
                                    std_percentage=args.std_percentage, augmentations=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

   

    # print number of datapoints
    print("Number of test datapoints:", test_dataset.X.shape[0])

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
                    dropout=args.dropout)
    elif args.model_type == "mlp":
        model = MLP(input_size=len(INPUT_FEATURES), 
                    output_size=len(OUTPUT_FEATURES),
                    num_layers=args.num_layers, 
                    dropout=args.dropout)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    
    batch = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing")
    model.eval()

    Y = np.zeros((test_dataset.X.shape[0], test_dataset.Y.shape[1]))
    Y_hat = np.zeros((test_dataset.X.shape[0], test_dataset.Y.shape[1]))

    # Inference speed computation
    with torch.no_grad():
        test_x, test_y = next(iter(test_dataloader))
        print(test_x.shape, test_y.shape)
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

    with torch.no_grad():

        for i, (x, y) in enumerate(batch):
            x = x.to(args.device).float()
            y = y.to(args.device).float()
            y_pred = model(x)
            Y[i*args.batch_size:(i+1)*args.batch_size, :] = y.cpu().numpy()
            Y_hat[i*args.batch_size:(i+1)*args.batch_size, :] = y_pred.cpu().numpy()
    
    # Plotting on pdf file
    
    # Y = Y[::50, :]
    # Y_hat = Y_hat[::50, :]

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


    
    
