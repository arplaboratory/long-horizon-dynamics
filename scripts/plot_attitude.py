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
    "save": ["phi", "theta", "psi"],
    "title": ["Ground velocity u", "Ground velocity v", "Ground velocity w", "Ground angular velocity p", "Ground angular velocity q", "Ground angular velocity r"], 
    # "test": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "qx", "qy", "qz", "qw",  "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"],
}

def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['inputs'][:]
        Y = hf['outputs'][:]
    return X, Y


def Quaternion2Euler(quaternion):
    """
    converts a quaternion attitude to an euler angle attitude
    :param quaternion: the quaternion to be converted to euler angles in a np.matrix
    :return: the euler angle equivalent (phi, theta, psi) in a np.array
    """
    e0 = quaternion[0, 0]
    e1 = quaternion[0, 1]
    e2 = quaternion[0, 2]
    e3 = quaternion[0, 3]
    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
    theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)

    return phi, theta, psi

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
    args.unroll_length = 60
    
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

    Y_plot = np.zeros((args.unroll_length,     3))
    Y_hat_plot = np.zeros((args.unroll_length, 3))

    model.eval()
    with torch.no_grad():

        x = X[2500, :, :]
        y = Y[2500, :, :]

        x = x.unsqueeze(0)
        x_curr = x 

        for j in range(args.unroll_length):
            
            y_hat = model.forward(x_curr, init_memory=True if j == 0 else False)
            attitude_gt = y[j, 3:7]

            # y_hat = attitude_gt + torch.randn(4).to(args.device) * 0.01
            # y_hat = y_hat.unsqueeze(0)

            # Normalize the quaternion
            y_hat = y_hat / torch.norm(y_hat, dim=1, keepdim=True)

 
            # Save predictions and ground truth attitude for plotting
            # Convert quaternion to euler angles
            euler_pred = Quaternion2Euler(y_hat.detach().cpu().numpy())
            
            euler_gt = Quaternion2Euler(attitude_gt.unsqueeze(0).detach().cpu().numpy())

            Y_plot[j, :] = euler_gt
            Y_hat_plot[j, :] = euler_pred
            
            
            if j < args.unroll_length - 1:
                
                linear_velocity_gt = y[j, :3].view(1, 3)
                angular_velocity_gt = y[j, 7:10].view(1, 3)

                u_gt = y[j, -4:].unsqueeze(0)
                # Update x_curr
                x_unroll_curr = torch.cat((linear_velocity_gt, y_hat, angular_velocity_gt, u_gt), dim=1)
            
                x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)


    #################################################################################################################################################
                
    # Plot the predicted and ground truth Quaternion
    for i in range(3):
        fig = plt.figure(figsize=(8, 6), dpi=400)

        plt.ylim(-1, 1)


        plt.plot(Y_plot[:, i], label="Ground Truth", color=colors[1], linewidth=4.5)
        plt.plot(Y_hat_plot[:, i], label="Predicted", color=colors[2], linewidth=4.5, linestyle=line_styles[1])
        plt.grid(True)  # Add gridlines
        plt.tight_layout(pad=1.5)
        plt.legend()
        plt.xlabel("No. of recursive predictions")
        plt.ylabel(OUTPUT_FEATURES["euler"][i])
        # plt.title(OUTPUT_FEATURES["title"][i])
        
        plt.savefig(experiment_path + "plots/trajectory/trajectory_" + OUTPUT_FEATURES["save"][i] + ".png")
        plt.close()    
                
    #################################################################################################################################################

