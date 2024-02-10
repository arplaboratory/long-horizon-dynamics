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
    "save": ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"],
    "title": ["Ground velocity u", "Ground velocity v", "Ground velocity w", "Ground angular velocity p", "Ground angular velocity q", "Ground angular velocity r"], 
    # "test": ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "qx", "qy", "qz", "qw",  "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"],
}

def load_data(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as hf: 
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
    plotting_data_path = experiment_path + "plotting_data/"

    check_folder_paths([os.path.join(experiment_path, "checkpoints"), os.path.join(experiment_path, "plots"), os.path.join(experiment_path, "plots", "trajectory"), 
                        os.path.join(experiment_path, "plots", "testset"), plotting_data_path])

    print(experiment_path)
    print("Testing Dynamics model:", model_path)
    args = load_args(experiment_path + "args.txt")

    model_name = args.model_type + "_" + str(args.history_length) + "_" + args.vehicle_type + "_" + str(args.unroll_length) 

    args.unroll_length = 60
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    # Load all hdf5 paths in the data folder into a list sorted 
    hdf5_files = sorted(glob.glob(data_path + "test/*.h5"))

    X_test = []
    Y_test = []
    trajectory_name = []

    for hdf5_file in hdf5_files:
        

        # create the dataset
        X, Y = load_data(hdf5_file)

        # convert X and Y to tensors
        X = torch.from_numpy(X).float().to(args.device)
        Y = torch.from_numpy(Y).float().to(args.device)

        X_test.append(X)
        Y_test.append(Y)

        trajectory_name.append(hdf5_file.split("/")[-1].split(".")[0])
    
        
    print(trajectory_name)
    print('Loading model ...')

    # Print vehicle type
    print("Vehicle Type: ", args.vehicle_type)

    # Initialize the model
    
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=X_test[0].shape[-1],
        output_size=6,
        valid_data=Y_test[0],
        max_iterations=1,
    )

    # # Load the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(args.device)

    input_shape = (1, args.history_length, X_test[0].shape[-1])
    output_shape = (1, 6)

    mse_loss = MSE()


    # Store the results for each trajectory in a dictionary
    results = {}        

    # Loop through the test data using tqdm
    for t, (X, Y) in enumerate(tqdm(zip(X_test, Y_test), total=len(X_test), desc="Testing")):

        sample_loss = []
        copounding_error_per_sample = []
        mean_abs_error_per_sample = []  

        model.eval()
        with torch.no_grad():

            for i in tqdm(range(0, X.shape[0])):

                x = X[i, :, :]
                y = Y[i, :, :]

                x = x.unsqueeze(0)
                x_curr = x 
                batch_loss = 0.0

                abs_error = []

                compounding_error = []

                for j in range(args.unroll_length):
                    y_hat = model.forward(x_curr, init_memory=True if j == 0 else False)

                    linear_velocity_gt =  y[j, :3]
                    angular_velocity_gt = y[j, 7:10]

                    velocity_gt = torch.cat((linear_velocity_gt, angular_velocity_gt), dim=0)

                    abs_error.append(torch.abs(y_hat - velocity_gt))

                    loss = mse_loss(y_hat, velocity_gt)
                    batch_loss += loss / args.unroll_length

                    compounding_error.append(loss.item())

                    if j < args.unroll_length - 1:

                        linear_velocity_pred = y_hat[:, :3]
                        angular_velocity_pred = y_hat[:, 3:]

                        u_gt = y[j, -4:].unsqueeze(0)
                        attitude_gt = y[j, 3:7].unsqueeze(0)

                        # Update x_curr
                        x_unroll_curr = torch.cat((linear_velocity_pred, attitude_gt, angular_velocity_pred, u_gt), dim=1)
                    
                        x_curr = torch.cat((x_curr[:, 1:, :], x_unroll_curr.unsqueeze(1)), dim=1)
                
                mean_abs_error_per_sample.append(torch.mean(torch.cat(abs_error, dim=0), dim=0).cpu().numpy())
                sample_loss.append(batch_loss.item())
                copounding_error_per_sample.append(compounding_error)

        # Save compound error per sample as numpy array with trajectory name and model name
        np.save(plotting_data_path + trajectory_name[t] + "_" + model_name + ".npy", np.array(copounding_error_per_sample))

        # Save Mean Copounding Error per sample, variance Copounding Error per sample, mean_abs_error_per_sample, and mean_sample_loss in a dictionary
        results[trajectory_name[t]] = {
            "mean_compounding_error": np.mean(copounding_error_per_sample, axis=0),
            "variance_compounding_error": np.var(copounding_error_per_sample, axis=0),
            "mean_abs_error_per_sample": np.mean(mean_abs_error_per_sample, axis=0),
            "mean_sample_loss": np.mean(sample_loss)
        }

        print("Mean Sample Loss: ", np.mean(sample_loss))

        mean_copounding_error_per_sample = np.mean(copounding_error_per_sample, axis=0)
        var_copounding_error_per_sample = np.var(copounding_error_per_sample, axis=0)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.plot(mean_copounding_error_per_sample, color='skyblue', linewidth=2.5, label='Mean Copounding Error')
        ax.fill_between(np.arange(len(mean_copounding_error_per_sample)), mean_copounding_error_per_sample - var_copounding_error_per_sample, 
                        mean_copounding_error_per_sample + var_copounding_error_per_sample, alpha=0.5, color='skyblue', 
                        label='Variance Copounding Error')

        ax.set_xlabel("No. of Recursive Predictions")
        ax.set_ylabel("MSE")
        ax.set_title("MSE Analysis over Recursive Predictions")
        ax.legend()

        # Save the plot
        plt.tight_layout(pad=1.5)
        plt.savefig(experiment_path + "plots/trajectory/" + trajectory_name[t] + "_" + model_name + ".pdf")
        plt.close()


        # Print the mean_abs_error_per_sample over the entire test set
        mean_abs_error_per_sample = np.array(mean_abs_error_per_sample)
        print("Mean Absolute Error per sample: ", np.mean(mean_abs_error_per_sample, axis=0))
        
        # Plot sample loss as bar plot over every sample 
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.bar(np.arange(len(sample_loss)), sample_loss, color='skyblue', alpha=0.7, edgecolor='black')

        ax.set_xlabel("Sample")
        ax.set_ylabel("MSE")
        # Set title with the unroll length
        ax.set_title(f"MSE over {args.unroll_length} Recursive Predictions")

        # Adding gridlines
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Customize ticks and labels
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='x', rotation=45)

        # Save the plot
        plt.tight_layout(pad=1.5)
        plt.savefig(experiment_path + "plots/trajectory/" + trajectory_name[t] + "_" + model_name + "_sample_loss.pdf")
        plt.close()

# Print final results
print("Final Results: ", results)
np.save(plotting_data_path + trajectory_name[t] + "_" + model_name + "_results" + ".npy", results)



        


        


             


        
        
        