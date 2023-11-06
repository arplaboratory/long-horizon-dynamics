import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import scipy.linalg as linalg
import h5py
from utils import Euler2Quaternion, Quaternion2Euler, Euler2Rotation, Quaternion2Rotation, Rotation2Quaternion

class DynamicsDataset(Dataset):
    def __init__(self, data_path, hdf5_file, args):

        self.normalize = args.normalize
        self.X, self.Y = self.load_data(data_path, hdf5_file)
        self.history_length = args.history_length
        self.unroll_length = args.unroll_length
        self.batch_size = args.batch_size
        self.num_steps = np.ceil(self.X.shape[0] / self.batch_size).astype(int)
        self.augmentations = args.augmentation
        self.std_percentage = args.std_percentage
        self.attitude = args.attitude
        self.model_type = args.model_type
        
        
        if self.history_length == 0:
            assert self.X.shape[1] == self.Y.shape[2]
            assert self.X.shape[0] == self.Y.shape[0]  + 4
            self.X_shape = self.X.shape
            self.Y_shape = self.Y.shape

        else:
            assert self.X.shape[2] == self.Y.shape[2]
            assert self.X.shape[0] == self.Y.shape[0]
            self.X_shape = self.X.shape
            self.Y_shape = self.Y.shape

    def __len__(self):

        return self.X.shape[2]

    def __getitem__(self, idx):

        if self.history_length == 0:
            x = self.X[:, idx]
            y = self.Y[:, idx]

        else:
           
            if self.model_type == 'mlp':
                x = self.X[:, :, idx].flatten('F')
            else:
                x = self.X[:, :, idx]
            y = self.Y[:, :, idx]
            
        if self.augmentations and random.random() < 0.8:
            x = self.augment(x)
        
        return x.T, y
        
    def load_data(self, hdf5_path, hdf5_file):
        with h5py.File(os.path.join(hdf5_path, hdf5_file), 'r') as hf: 
            X = hf['X'][:]
            Y = hf['Y'][:]
        return X.T, Y.T
    
    def augment(self, x):
        """
        Augments the input data with noise
        :param x: input data
        :return: augmented data
        """


        if self.history_length == 0:

            if self.attitude == 'quaternion':

                attitude_euler = Quaternion2Euler(x[3:7])
                attitude_euler += np.random.normal(0, self.std_percentage * np.abs(attitude_euler))
                x[3:7] = Euler2Quaternion(attitude_euler[0], attitude_euler[1], attitude_euler[2]).T

                for feature_index in range(self.X.shape[0]):
                    if feature_index < 3 or feature_index > 6:
                        noise_std = self.std_percentage * np.abs(x[feature_index])

                        # Add noise to the input feature
                        x[feature_index] += np.random.normal(0, noise_std)

            elif self.attitude == 'rotation':
                
                attitude_quaternion = Rotation2Quaternion(x[3:12].reshape(3, 3))
                attitude_euler = Quaternion2Euler(attitude_quaternion)
            
                attitude_euler += np.random.normal(0, self.std_percentage * np.abs(attitude_euler))
                attitude_quaternion = Euler2Quaternion(attitude_euler[0], attitude_euler[1], attitude_euler[2]).T
                x[3:12] = Quaternion2Rotation(attitude_quaternion).flatten()

                # print determinant of rotation matrix
                
                for feature_index in range(self.X.shape[0]):
                    if feature_index < 3 or feature_index > 11:
                        noise_std = self.std_percentage * np.abs(x[feature_index])

                        # Add noise to the input feature
                        x[feature_index] += np.random.normal(0, noise_std)

            elif self.attitude == 'euler':
                for feature_index in range(self.X.shape[0]):
                    # Do not add noise to the attitude
                    noise_std = self.std_percentage * np.abs(x[feature_index])

                    # Add noise to the input feature
                    x[feature_index] += np.random.normal(0, noise_std)
        else:

            if self.attitude == 'quaternion':
                
                for t in range(self.history_length):
                    # Add noise to the attitude
                    attitude_euler = Quaternion2Euler(x[3:7, t])
                    attitude_euler += np.random.normal(0, self.std_percentage * np.abs(attitude_euler))
                    x[3:7, t] = Euler2Quaternion(attitude_euler[0], attitude_euler[1], attitude_euler[2]).T

                    for feature_index in range(self.X.shape[0]):
                        if feature_index < 3 or feature_index > 6:
                            noise_std = self.std_percentage * np.abs(x[feature_index, t])
                            
                            # Add noise to the input feature
                            x[feature_index, t] += np.random.normal(0, noise_std)
            
            elif self.attitude == 'rotation':
                feature_index = [0, 1, 2, 12, 13, 14]  # Modify this list with the desired indices

                for t in range(0, self.history_length):
                    attitude_quaternion = Rotation2Quaternion(x[3:12, t].reshape(3, 3))
                    attitude_euler = Quaternion2Euler(attitude_quaternion)
                    
                    # Precompute constants
                    noise_std_attitude = self.std_percentage * np.abs(attitude_euler)
                    
                    # Generate random noise for all elements at once
                    noise = np.random.normal(0, noise_std_attitude)
                    
                    # Add noise to attitude quaternion
                    attitude_euler += noise
                    
                    # Convert back to quaternion
                    attitude_quaternion = Euler2Quaternion(attitude_euler[0], attitude_euler[1], attitude_euler[2]).T
                    
                    # Apply quaternion to rotation matrix transformation
                    attitude_rotation = Quaternion2Rotation(attitude_quaternion)
                    
                    # Replace the entire column with the updated values
                    x[3:12, t] = attitude_rotation.flatten()
                    
                    # Apply noise to elements where feature_index < 3 or feature_index > 11
                    noise_std_features = self.std_percentage * np.abs(x[:3, t])
                    x[:3, t] += np.random.normal(0, noise_std_features)
                    
                    # Apply noise to elements specified by feature_index
                    noise_std_features = self.std_percentage * np.abs(x[feature_index, t])
                    x[feature_index, t] += np.random.normal(0, noise_std_features)
            
            elif self.attitude == 'euler':
                for t in range(self.history_length):
                    for feature_index in range(self.X.shape[0]):
                        # Do not add noise to the attitude
                        noise_std = self.std_percentage * np.abs(x[feature_index, t])
                        
                        # Add noise to the input feature
                        x[feature_index, t] += np.random.normal(0, noise_std)
        return x
