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

# Function to recursively find linear layers and save their weight distribution plots
def find_and_save_linear_layers(module, module_name, save_dir):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Extract the weights
            weights = child.weight.data.cpu().numpy().flatten()

            # Create a histogram
            plt.figure(figsize=(8, 6))
            n, bins, patches = plt.hist(weights, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

            # Add labels and title
            plt.title(f'Weight Distribution - {module_name} ({name})', fontsize=16)
            plt.xlabel('Weight Value', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)

            # Add a grid
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Add a legend
            plt.legend([f'Mean: {np.mean(weights):.2f}', f'Std Dev: {np.std(weights):.2f}'], fontsize=12)

            # Save the plot as an image file
            save_path = os.path.join(save_dir, f'weight_distribution_{module_name}_{name}.png')
            plt.savefig(save_path, bbox_inches='tight')

        else:
            # If not a linear layer, recursively check its children
            find_and_save_linear_layers(child, f"{module_name} ({name})", save_dir)


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
    args.device = "cpu"
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
                          'r11', 'r21', 'r31', 
                          'r12', 'r22', 'r32',
                          'r13', 'r23', 'r33',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'r11', 'r12', 'r13', 
                           'r21', 'r22', 'r23',
                           'r31', 'r32', 'r33',
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
    test_dataset = DynamicsDataset(data_path + "test/", 'test.h5', args.batch_size, normalize=args.normalize, 
                                    std_percentage=args.std_percentage, attitude=args.attitude, augmentations=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

   

    # print number of datapoints
    print("Number of test datapoints:", test_dataset.X.shape[1])

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
    model.load_state_dict(torch.load(model_path))

    # Save the weight distribution plots
    save_dir = experiment_path + "plots/"
    check_folder_paths([save_dir])
    find_and_save_linear_layers(model, "MLP", save_dir)

    
   