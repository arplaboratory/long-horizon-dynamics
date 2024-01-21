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


OUTPUT_FEATURES = {
    "euler": ["phi (rad)", "theta (rad)", "psi (rad)"],
    "quaternion": ["q0", "q1", "q2", "q3"],
    "rotation": ["u", "v", "w", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "p", "q", "r"],
    "test": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"], 
    "save": ["vx", "vy", "vz", "wx", "wy", "wz"],
    "title": ["Ground velocity u", "Ground velocity v", "Ground velocity w", "Ground angular velocity p", "Ground angular velocity q", "Ground angular velocity r"], 
    # "test": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "qx", "qy", "qz", "qw",  "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"],
}

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
        output_size=6,
        valid_data=Y,
        max_iterations=1,
    )

    # # Load the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(args.device)

    input_shape = (1, args.history_length, X.shape[-1])
    output_shape = (1, 6)

    Y_plot = np.zeros((args.unroll_length,     6))
    Y_hat_plot = np.zeros((args.unroll_length, 6))

    model.eval()
    with torch.no_grad():

        x = X[100, :, :]
        y = Y[100, :, :]

        x = x.unsqueeze(0)
        x_curr = x 

        for j in range(args.unroll_length):
            
            y_hat = model.forward(x_curr, init_memory=True if j == 0 else False)
            
            linear_velocity_gt =  y[j, :3]
            angular_velocity_gt = y[j, 7:10]

            velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=0)

        
            Y_plot[j, :] = velocity_gt.detach().cpu().numpy()
            Y_hat_plot[j, :] = y_hat.detach().cpu().numpy()
            
            
            if j < args.unroll_length - 1:
                
                linear_velocity_pred = y_hat[:, :3]
                angular_velocity_pred = y_hat[:, 3:]

                u_gt = y[j, -4:].unsqueeze(0)
                attitude_gt = y[j, 3:7].unsqueeze(0)

                # Update x_curr
                x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)
            
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)


    #################################################################################################################################################
                
    # Plot the predicted and ground truth Quaternion
    pp = PdfPages(experiment_path + "plots/trajectory/trajectory.pdf")

    # Plot linear velocity
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.suptitle("Linear Velocity", x=0.5, y=0.95)
    
    # zoom out the plot
    ax[0].set_ylim([-2.0, 2.0])
    ax[1].set_ylim([-2.0, 2.0])
    ax[2].set_ylim([-2.0, 2.0])
    

    ax[0].plot(Y_plot[:, 0], label="Ground Truth", color=colors[1], linewidth=4.5)
    ax[0].plot(Y_hat_plot[:, 0], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
    ax[1].plot(Y_plot[:, 1], label="Ground Truth", color=colors[1], linewidth=4.5)
    ax[1].plot(Y_hat_plot[:, 1], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
    ax[2].plot(Y_plot[:, 2], label="Ground Truth", color=colors[1], linewidth=4.5)
    ax[2].plot(Y_hat_plot[:, 2], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
    ax[0].set_ylabel(r'$v_x$ $[m/s]$')
    ax[1].set_ylabel(r'$v_y$ $[m/s]$')
    ax[2].set_ylabel(r'$v_z$ $[m/s]$')
    ax[2].set_xlabel("No. of recursive predictions")
    ax[0].grid(); ax[1].grid(); ax[2].grid()
    fig.align_ylabels(ax)
    pp.savefig(fig)

    # Plot angular velocity
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.suptitle("Angular Velocity", x=0.5, y=0.95)
    # zoom out the plot
    ax[0].set_ylim([-2.0, 2.0])
    ax[1].set_ylim([-2.0, 2.0])
    ax[2].set_ylim([-2.0, 2.0])
    
    ax[0].plot(Y_plot[:, 3], label="Ground Truth", color=colors[1], linewidth=4.5)
    ax[0].plot(Y_hat_plot[:, 3], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
    ax[1].plot(Y_plot[:, 4], label="Ground Truth", color=colors[1], linewidth=4.5)
    ax[1].plot(Y_hat_plot[:, 4], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
    ax[2].plot(Y_plot[:, 5], label="Ground Truth", color=colors[1], linewidth=4.5)
    ax[2].plot(Y_hat_plot[:, 5], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
    ax[0].set_ylabel(r'$\omega_x$')
    ax[1].set_ylabel(r'$\omega_y$')
    ax[2].set_ylabel(r'$\omega_z$')
    ax[2].set_xlabel("No. of recursive predictions")
    ax[0].grid(); ax[1].grid(); ax[2].grid()
    fig.align_ylabels(ax)
    pp.savefig(fig)

    pp.close()
    plt.close("all")
                
    #################################################################################################################################################

