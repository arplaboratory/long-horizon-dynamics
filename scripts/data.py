import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
import random


class DynamicsDataset(Dataset):
    def __init__(self, data_path, batch_size, input_features, 
                 output_features, history_length, normalize, std_percentage, 
                 augmentations=True):

        self.normalize = normalize
        self.mean = np.zeros((len(output_features), 1))
        self.std = np.ones((len(output_features), 1))
        self.history_length = history_length
        self.X, self.Y = self.load_data(data_path, input_features, output_features)
        self.batch_size = batch_size
        self.num_steps = np.ceil(self.X.shape[0] / self.batch_size).astype(int)
        self.augmentations = augmentations
        self.std_percentage = std_percentage
        
        assert self.X.shape[0] == self.Y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        x = self.X[idx, :, :]
        y = self.Y[idx, :]


        if self.augmentations and random.random() < 0.5:

            for t in range(self.history_length):
                for feature_index in range(self.X.shape[2]):
                    noise_std = self.std_percentage * np.abs(x[t, feature_index])
                    
                    # Add noise to the input feature
                    x[t, feature_index] += np.random.normal(0, noise_std)

        return x, y
        
    def load_data(self, data_path, input_features, output_features):
        """
        Read data from multiple CSV files in a folder and prepare concatenated input-output pairs.

        Args:
        - folder_path (str): Path to the folder containing CSV files.
        - input_features (list): List of feature names to include in the input array.
        - output_features (list): List of feature names to include in the output array.

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

                num_samples = len(data) - self.history_length
                num_input_features = len(input_features)
                num_output_features = len(output_features)

                X = np.zeros((num_samples, self.history_length, num_input_features))
                Y = np.zeros((num_samples, num_output_features))

                for i in range(num_samples):
                    for j in range(self.history_length):
                        for k in range(num_input_features):
                            X[i, j, k] = float(data[i + j][input_features[k]])
                    for k in range(num_output_features):
                        Y[i, k] = float(data[i + self.history_length][output_features[k]])
                
                all_X.append(X)
                all_Y.append(Y)
        
        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)

        return X, Y
    

    
   