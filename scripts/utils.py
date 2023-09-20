import os
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


INPUT_FEATURES = ['u', 'v', 'w',
                      'r11', 'r21', 'r31', 
                      'r12', 'r22', 'r32',
                      'p', 'q', 'r',
                      'delta_e', 'delta_a', 'delta_r', 'delta_t']
OUTPUT_FEATURES = ['u', 'v', 'w',
                    'p', 'q', 'r',]

def load_data(data_path, input_features, output_features, use_history=False, history_length=4):
    """
    Read data from multiple CSV files in a folder and prepare concatenated input-output pairs.

    Args:
    - data_path (str): Path to the folder containing CSV files.
    - input_features (list): List of feature names to include in the input array.
    - output_features (list): List of feature names to include in the output array.
    - use_history (bool): If True, include historical state-action pairs in the input; if False, use only the current state-action pair.
    - history_length (int): Number of historical state-action pairs to consider if use_history is True.

    Returns:
    - X (numpy.ndarray): Concatenated input array with selected features.
    - Y (numpy.ndarray): Concatenated output array with selected features.
    """

    all_X = []
    all_Y = []

    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(data_path, filename)
            
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                data = [row for row in reader]

            num_samples = len(data) - (history_length if use_history else 1)
            num_input_features = len(input_features)
            num_output_features = len(output_features)

            if use_history:
                X = np.zeros((num_samples, history_length, num_input_features))
            else:
                X = np.zeros((num_samples, num_input_features))
                
            Y = np.zeros((num_samples, num_output_features))

            for i in range(num_samples):
                if use_history:
                    for j in range(history_length):
                        for k in range(num_input_features):
                            X[i, j, k] = float(data[i + j][input_features[k]])
                else:
                    for k in range(num_input_features):
                        X[i, k] = float(data[i][input_features[k]])

                for k in range(num_output_features):
                    Y[i, k] = float(data[i + (history_length if use_history else 1)][output_features[k]])

            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    return X, Y

def plot_data(data, features, save_path):
    """Plot data

    Args:
        data (ndarray): data
        features (List[str]): list of features
        save_path (str, optional): path to save figure. Defaults to None.
    """
    for i, feature in enumerate(features):
        plt.figure(i)
        plt.plot(data[i, :])
        plt.title(feature)
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel(feature)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, feature+'.png'))


def check_folder_paths(folder_paths):
    for path in folder_paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Creating folder", path, "...")

if __name__=="__main__":


    x, y = load_data('/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/train', INPUT_FEATURES, OUTPUT_FEATURES, history_length=4, use_history=True)
    print(x.shape, y.shape)

    print(x[0, :])
    print(y[0, :])

    # print(data)
    # print(x.shape, y.shape)
    # plot_data(x, features, '/home/prat/arpl/TII/ws_dynamics/data/train')


