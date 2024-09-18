import os
import pandas as pd
import numpy as np
import h5py
import sys
from config import parse_args
import glob

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib


plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans",
    "font.size": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

colors = ["#365282", "#E84C53", "#9FBDA0","#edb120", "#ff7f00", "#984ea3", "#f781bf", "#999999"]
markers = ['o', 's', '^', 'D', 'v', 'p']
line_styles = ['-', '--', '-.', ':', '-', '--']

# load hdf5
def load_hdf5(data_path, hdf5_file):
    with h5py.File(data_path + hdf5_file, 'r') as hf:
        X = hf['inputs'][:]
        Y = hf['outputs'][:]

    return X, Y


# Numpy function to compute MSE 
def MSE(y_true, y_pred):
    
    loss = (y_true - y_pred)**2
    loss = loss.sum(axis=1)

    return loss



if __name__ == "__main__":
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
    data_path = resources_path + "data/" + args.dataset + "/"

    # load the saved trajectory_error.npy
    trajectory_error = np.load(experiment_path + "plots/trajectory_error.npy")

    # Load test csv in the test folder
    test_data_path = resources_path + "data/" + args.dataset + "/test/"
    test_csv = [f for f in os.listdir(test_data_path) if f.endswith('.csv')][0]

    # Load the test csv file
    df = pd.read_csv(test_data_path + test_csv)

    # Load the test hdf5 file
    hdf5_files = [f for f in os.listdir(test_data_path) if f.endswith('.h5')]
    X, Y = load_hdf5(test_data_path, hdf5_files[0])

    # Get MSE between the last input and the first output for linear and angular velocities
    MSE_linear_velocities = MSE(X[:,-1,0:3], Y[:,0,0:3])
    MSE_angular_velocities = MSE(X[:,-1,7:10], Y[:,0,7:10])

    total_MSE = MSE_linear_velocities + MSE_angular_velocities
    total_MSE = total_MSE.reshape(-1, 1)


    # Plot the trajectory error and the MSE between the the last input and the first output for linear and angular velocities on the same plot
    # In another subplot, plot the wind speed 

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot the trajectory error
    axs[0].plot(trajectory_error, color=colors[0], label="Prediction MSE")
    axs[0].plot(total_MSE, color=colors[1], label="Ground Truth Delta MSE")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Sample")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Plot the wind speed
    axs[1].plot(df["wind_north"], color=colors[2], label="Wind North")
    axs[1].plot(df["wind_east"], color=colors[3], label="Wind East")
    axs[1].set_ylabel("Wind Speed")
    axs[1].set_xlabel("Sample")
    axs[1].legend() 
    axs[1].grid(alpha=0.3)

    # Plot angular velocities
    axs[2].plot(df["ang_vel_x"], color=colors[4], label="Angular Velocity X")
    axs[2].plot(df["ang_vel_y"], color=colors[5], label="Angular Velocity Y")
    axs[2].plot(df["ang_vel_z"], color=colors[6], label="Angular Velocity Z")

    axs[2].set_ylabel("Angular Velocity")
    axs[2].set_xlabel("Sample")
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.tight_layout()

    # Save plot in the experiment folder as pdf
    plt.savefig(experiment_path + "plots/trajectory_error.pdf")
    


    

    
