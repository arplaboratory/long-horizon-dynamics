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

    features = ['u', 'v', 'w',
                'phi', 'theta', 'psi', 
                'p', 'q', 'r', 
                'delta_e', 'delta_a', 'delta_r', 'delta_t']

    # create the dataseT
    test_dataset = DynamicsDataset(data_path + "test/", 1, features, args.normalize)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=args.shuffle, num_workers=args.num_workers)
    
    # Initialize the model
    model = MLP(input_size=test_dataset.X.shape[0], 
                output_size=test_dataset.Y.shape[0],
                num_layers=[512, 512, 256, 128, 64], 
                dropout=args.dropout).to(args.device)
    

    # make predicitons and visualize outputs with ground truth
    model.load_state_dict(torch.load(model_path))
    batch = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing")

    model.eval()
    Y = np.zeros((test_dataset.Y.shape[1], test_dataset.Y.shape[0]))
    Y_hat = np.zeros((test_dataset.Y.shape[1], test_dataset.Y.shape[0]))

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
            torch.cuda.synchronize()
            if k:
                frequency_hist.append(1. / (starter.elapsed_time(ender) / 1000))
        print("Inference speed (Hz): ", np.mean(frequency_hist))

    with torch.no_grad():
        for i, (x, y) in enumerate(batch):
            x = x.to(args.device).float()
            y = y.to(args.device).float()
            y_hat = model(x)

            Y[i*args.batch_size:(i+1)*args.batch_size, :] = y.cpu().numpy()
            Y_hat[i*args.batch_size:(i+1)*args.batch_size, :] = y_hat.cpu().numpy()



    features = ['u (m/s)', 'v (m/s)', 'w (m/s)',
                'phi (radians)', 'theta (radians)', 'psi (radians)', 
                'p (radians/sec)', 'q (radians/sec)', 'r (radians/sec)']
   
    Y = Y[::50, :]
    Y_hat = Y_hat[::50, :]
    
    # plot the prediction and ground truth per second in a pdf
    # with PdfPages(experiment_path + "plots/predictions.pdf") as pdf:
    #     for i in range(9):
    #         plt.figure(i)
    #         plt.plot(Y[:, i], label="Ground Truth", color=colors[2])
    #         plt.plot(Y_hat[:, i], label="Prediction", color=colors[3])
    #         plt.grid()
    #         plt.xlabel('Time')
    #         plt.ylabel(features[i])
    #         plt.legend(loc ="upper right")
    #         pdf.savefig()
    #         plt.close()

    # plot the prediction and ground truth per second in in a single plot
    with PdfPages(experiment_path + "plots/predictions.pdf") as p







