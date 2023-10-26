import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import check_folder_paths, plot_data
from config import parse_args, load_args

import sys
import glob
import time
import os

import seaborn as sns

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


HEADERS_TO_PLOT = ["u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]

def plot_histograms_from_csv(folder_path, headers_to_plot):
    # Initialize empty lists to store data from all CSV files
    all_data = {header: [] for header in headers_to_plot}

    # Loop through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Extract the specified headers and append to the lists
            for header in headers_to_plot:
                if header in df.columns:
                    all_data[header].extend(df[header])

    # Set a custom color palette for Seaborn (blue)
    sns.set_palette("Blues")

    # Create subplots based on the number of headers to plot
    num_plots = len(headers_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    # Plot histograms for the specified headers using the aggregated data
    for i, header in enumerate(headers_to_plot):
        sns.histplot(all_data[header], bins=20, edgecolor='black', ax=axes[i])
        axes[i].set_title(f'{header} Histogram')
        axes[i].set_xlabel(header)
        axes[i].set_ylabel('Frequency')

        # Add a grid to the plots
        axes[i].grid(True, linestyle='--', alpha=0.6)

    # Customize the appearance further if needed

    # Display the histograms
    plt.tight_layout()
    # save the figure
    plt.savefig(folder_path + 'histograms.png')


def generate_lag_plots(file_path):


    data = pd.read_csv(file_path)

    # Set a custom color palette for Seaborn (blue)
    sns.set_palette("Reds")

    # Create subplots based on the number of headers to plot
    num_plots = len(HEADERS_TO_PLOT)

    # Generate lag plots for the specified headers using the aggregated data
    for i, header in enumerate(HEADERS_TO_PLOT):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.scatterplot(x=data[header].shift(1), y=data[header], ax=axes[0])
        axes[0].set_title(f'{header} Lag Plot (1)')
        axes[0].set_xlabel(f'{header} (t-1)')
        axes[0].set_ylabel(f'{header} (t)')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        sns.scatterplot(x=data[header].shift(2), y=data[header], ax=axes[1])
        axes[1].set_title(f'{header} Lag Plot (2)')
        axes[1].set_xlabel(f'{header} (t-2)')
        axes[1].set_ylabel(f'{header} (t)')
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # save the figure
        plt.savefig(file_path + f'{header}_lag_plots.png')
        plt.close()



if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"

    train_path = data_path + "train/"
    valid_path = data_path + "valid/"

    file_path = train_path + 'dataset_sinusoid_4.csv'

    headers_to_plot = ['u', 'v', 'w', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'delta_e', 'delta_a', 'delta_r', 'delta_t']

    # plot_histograms_from_csv(train_path, headers_to_plot)

    generate_lag_plots(file_path)