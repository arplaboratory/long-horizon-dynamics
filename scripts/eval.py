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
    assert args.vehicle_type in ["fixed_wing", "quadrotor", "neurobem"], "Vehicle type must be one of [fixed_wing, quadrotor, neurobem]"

    if args.vehicle_type == "fixed_wing":
        vehicle_type = "fixed_wing"
    elif args.vehicle_type == "quadrotor":
        vehicle_type = "quadrotor"
    elif args.vehicle_type == "neurobem":
        vehicle_type = "neurobem"

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

    Y_plot = np.zeros((X.shape[0],     4))
    Y_hat_plot = np.zeros((X.shape[0], 4))

    model.eval()
    with torch.no_grad():

        for i in tqdm(range(0, X.shape[0])):

            x = X[i, :, :]
            y = Y[i, 0, :]

            x = x.unsqueeze(0)

            # Single step prediction
            y_hat = model.forward(x, init_memory=True if i == 0 else False)

            # Normalize the predicted Quaternion
            y_hat = y_hat / torch.norm(y_hat)


            attitude_gt = y[3:7]

            Y_plot[i, :] = attitude_gt.detach().cpu().numpy()
            Y_hat_plot[i, :] = y_hat.detach().cpu().numpy()

    #################################################################################################################################################
                
    # Plot the predicted and ground truth Quaternion
    pp = PdfPages(experiment_path + "plots/trajectory/one_step.pdf")

    # Plot the predicted and ground truth Quaternion
    fig, ax = plt.subplots(nrows=4, ncols=1)
    fig.suptitle("One step prediction", fontsize=32)
   

    # Plot the ground truth
    ax[0].plot(Y_plot[:, 0], color=colors[1], label="Ground truth", linewidth=3)
    ax[1].plot(Y_plot[:, 1], color=colors[1], label="Ground truth", linewidth=3)
    ax[2].plot(Y_plot[:, 2], color=colors[1], label="Ground truth", linewidth=3)
    ax[3].plot(Y_plot[:, 3], color=colors[1], label="Ground truth", linewidth=3)

    # Plot the predicted
    ax[0].plot(Y_hat_plot[:, 0], color=colors[2], label="Predicted", linewidth=3)
    ax[1].plot(Y_hat_plot[:, 1], color=colors[2], label="Predicted", linewidth=3)
    ax[2].plot(Y_hat_plot[:, 2], color=colors[2], label="Predicted", linewidth=3)
    ax[3].plot(Y_hat_plot[:, 3], color=colors[2], label="Predicted", linewidth=3)

    # Set the labels
    ax[0].set_ylabel(r"$q_w$", fontsize=32)
    ax[1].set_ylabel(r"$q_x$", fontsize=32)
    ax[2].set_ylabel(r"$q_y$", fontsize=32)
    ax[3].set_ylabel(r"$q_z$", fontsize=32)

    ax[3].set_xlabel("Time $[s]$", fontsize=32)

    # Set the legend
    ax[0].legend(loc="upper right", fontsize=32)
    ax[1].legend(loc="upper right", fontsize=32)
    ax[2].legend(loc="upper right", fontsize=32)
    ax[3].legend(loc="upper right", fontsize=32)

    pp.savefig(fig)

    
    pp.close()
    plt.close("all")
                
    #################################################################################################################################################
