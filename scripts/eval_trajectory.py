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
    "euler": ["u", "v", "w", "phi", "theta", "psi", "p", "q", "r"],
    "quaternion": ["u", "v", "w", "q0", "q1", "q2", "q3", "p", "q", "r"],
    "rotation": ["u", "v", "w", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33", "p", "q", "r"],
    "test": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"], 
    "title": ["Ground velocity u", "Ground velocity v", "Ground velocity w", "Ground angular velocity p", "Ground angular velocity q", "Ground angular velocity r"], 
    "test": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "qx", "qy", "qz", "qw",  "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"],
}


def load_data(hdf5_path, hdf5_file):
    with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
        X = hf['inputs'][:]
        Y = hf['outputs'][:]
    return X, Y

def Quaternion2Rotation(quaternion):
    """
    converts a quaternion attitude to a rotation matrix
    """
    e0 = quaternion[0]
    e1 = quaternion[1]
    e2 = quaternion[2]
    e3 = quaternion[3]

    R = torch.tensor([
        [e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
        [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
        [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]
    ])

    R = R / torch.det(R)

    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    tmp = r11 + r22 + r33
    if tmp > 0:
        e0 = 0.5 * torch.sqrt(1 + tmp)
    else:
        e0 = 0.5 * torch.sqrt(((r12 - r21) ** 2 + (r13 - r31) ** 2 + (r23 - r32) ** 2) / (3 - tmp))

    tmp = r11 - r22 - r33
    if tmp > 0:
        e1 = 0.5 * torch.sqrt(1 + tmp)
    else:
        e1 = 0.5 * torch.sqrt(((r12 + r21) ** 2 + (r13 + r31) ** 2 + (r23 - r32) ** 2) / (3 - tmp))

    tmp = -r11 + r22 - r33
    if tmp > 0:
        e2 = 0.5 * torch.sqrt(1 + tmp)
    else:
        e2 = 0.5 * torch.sqrt(((r12 + r21) ** 2 + (r13 + r31) ** 2 + (r23 + r32) ** 2) / (3 - tmp))

    tmp = -r11 - r22 + r33
    if tmp > 0:
        e3 = 0.5 * torch.sqrt(1 + tmp)
    else:
        e3 = 0.5 * torch.sqrt(((r12 - r21) ** 2 + (r13 + r31) ** 2 + (r23 + r32) ** 2) / (3 - tmp))

    unit_quaternion = torch.tensor([e0, e1, e2, e3])
    return unit_quaternion

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
    assert args.vehicle_type in ["fixed_wing", "quadrotor"], "Vehicle type must be one of [fixed_wing, quadrotor]"

    if args.vehicle_type == "fixed_wing":
        vehicle_type = "fixed_wing"
    else:
        vehicle_type = "quadrotor"

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
    X, Y = load_data(data_path + "test/", 'test_eval.h5')

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
        output_size=X.shape[-1]-4,
        valid_data=Y,
        max_iterations=1,
    )

    # Load the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(args.device)

    # load mean and std
    # mean = torch.from_numpy(np.load(data_path  + 'train/' + 'mean_train.npy')).float().to(args.device)
    # std =  torch.from_numpy(np.load(data_path  + 'train/' + 'std_train.npy')).float().to(args.device)

    input_shape = (1, args.history_length, X.shape[-1])
    output_shape = (1, X.shape[-1] - 4)

    input_tensor = X[0:args.history_length].reshape(input_shape)  


    Y_plot = np.zeros((X.shape[0] - args.history_length,     X.shape[-1] - 4))
    Y_hat_plot = np.zeros((X.shape[0] - args.history_length, X.shape[-1] - 4))

    mse_loss = MSE()
    trajectory_loss = []

    with torch.no_grad():
       for i in range(args.history_length -1, X.shape[0]):
        
            y_hat = model(input_tensor).view(output_shape) 

            # Normlaize the quaternion
            qw = y_hat[:, 3]
            qx = y_hat[:, 4]
            qy = y_hat[:, 5]
            qz = y_hat[:, 6]

            norm = torch.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2).view(-1, 1)
            y_hat[:, 3] = qw / norm
            y_hat[:, 4] = qx / norm
            y_hat[:, 5] = qy / norm
            y_hat[:, 6] = qz / norm

                    
            x_curr = torch.cat((y_hat, Y[i, -4:].unsqueeze(dim=0)), dim=1)
            input_tensor = torch.cat((input_tensor[:, 1:, :], x_curr.view(1, 1, X.shape[-1])), dim=1)

            if i < X.shape[0] :

                Y_plot[i - args.history_length, :] = Y[i, :-4].cpu().numpy()
                Y_hat_plot[i - args.history_length, :] = y_hat.cpu().numpy()

                # # PRINT mse loss
                
                # loss = mse_loss(y_hat, Y[i, :3].view(output_shape))
                loss = mse_loss(y_hat, Y[i, :-4].view(output_shape))
                trajectory_loss.append(loss.item())

    
    # Consider only the first 100 predictions 
    
    Y_plot = Y_plot[:10, :]
    Y_hat_plot = Y_hat_plot[:10, :]
    trajectory_loss = trajectory_loss[:10]


    
    mean_loss = np.mean(trajectory_loss)
    print("MSE Loss: ", mean_loss)
    
    # Y_plot = Y_plot[::50, :]
    # Y_hat_plot = Y_hat_plot[::50, :]


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
    plt.tight_layout(pad=1.5)
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
    plt.tight_layout(pad=1.5)
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
    for i in range(10):
        fig = plt.figure(figsize=(8, 6), dpi=400)
        plt.plot(Y_plot[:, i],     label="Ground Truth", color=colors[1], linewidth=4.5)
        plt.plot(Y_hat_plot[:, i], label="Predicted", color=colors[2], linewidth=4.5,  linestyle=line_styles[1])
        
        plt.grid(True)  # Add gridlines
        plt.tight_layout(pad=1.5)
        plt.legend()
        plt.xlabel("No. of recursive predictions")
        plt.ylabel(OUTPUT_FEATURES["test"][i])
        plt.savefig(experiment_path + "plots/trajectory/trajectory_" + OUTPUT_FEATURES['quaternion'][i] + ".png")
        plt.close()