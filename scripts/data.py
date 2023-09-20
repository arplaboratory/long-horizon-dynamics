import os
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
import random
import scipy.linalg as linalg
import h5py

class DynamicsDataset(Dataset):
    def __init__(self, data_path, hdf5_file, batch_size, normalize, std_percentage,
                 attitude, augmentations=True):

        self.normalize = normalize
        self.X, self.Y = self.load_data(data_path, hdf5_file)
        self.batch_size = batch_size
        self.num_steps = np.ceil(self.X.shape[0] / self.batch_size).astype(int)
        self.augmentations = augmentations
        self.std_percentage = std_percentage
        self.attitude = attitude
        
        assert self.X.shape[1] == self.Y.shape[1]

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):

        x = self.X[:, idx]
        y = self.Y[:, idx]
            
        if self.augmentations and random.random() < 0.5:

            if self.attitude == 'quaternion':
                attitude_euler = Quaternion2Euler(x[3:7])
                attitude_euler += np.random.normal(0, self.std_percentage * np.abs(attitude_euler))

                for feature_index in range(self.X.shape[0]):
                    
                    if feature_index < 3 or feature_index > 6:
                        # Do not add noise to the attitude
                        noise_std = self.std_percentage * np.abs(x[feature_index])
                        x[feature_index] += np.random.normal(0, noise_std)    
                noise_std = self.std_percentage * np.abs(x[feature_index])
                x[3:7] = Euler2Quaternion(attitude_euler[0], attitude_euler[1], attitude_euler[2]).T

            elif self.attitude == 'rotation':
                attitude_quaternion = Rotation2Quaternion(x[3:12].reshape(3,3))
                attitude_euler = Quaternion2Euler(attitude_quaternion)
                
                for feature_index in range(self.X.shape[0]):
                    if feature_index < 3 or feature_index > 11:
                        # Do not add noise to the attitude
                        noise_std = self.std_percentage * np.abs(x[feature_index])
                        x[feature_index] += np.random.normal(0, noise_std)
                attitude_euler += np.random.normal(0, self.std_percentage * np.abs(attitude_euler))
                attitude_quaternion = Euler2Quaternion(attitude_euler[0], attitude_euler[1], attitude_euler[2]).T
                # Convert the quaternion to a rotation matrix
                attitude_rotation = Quaternion2Rotation(attitude_quaternion)
                x[3:12] = attitude_rotation.flatten()

            elif self.attitude == 'euler':
                for feature_index in range(self.X.shape[0]):
                    # Do not add noise to the attitude
                    noise_std = self.std_percentage * np.abs(x[feature_index])
                    
                    # Add noise to the input feature
                    x[feature_index] += np.random.normal(0, noise_std)

        return x, y
        
    def load_data(self, hdf5_path, hdf5_file):
        with h5py.File(hdf5_path + hdf5_file, 'r') as hf: 
            X = hf['X'][:]
            Y = hf['Y'][:]
        return X.T, Y.T
    
def Quaternion2Euler(quaternion):
    """
    converts a quaternion attitude to an euler angle attitude
    :param quaternion: the quaternion to be converted to euler angles in a np.matrix
    :return: the euler angle equivalent (phi, theta, psi) in a np.array
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)
    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
    theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)

    return phi, theta, psi

def Rotation2Quaternion(R):
    """
    converts a rotation matrix to a unit quaternion
    """
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    tmp=r11+r22+r33
    if tmp>0:
        e0 = 0.5*np.sqrt(1+tmp)
    else:
        e0 = 0.5*np.sqrt(((r12-r21)**2+(r13-r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=r11-r22-r33
    if tmp>0:
        e1 = 0.5*np.sqrt(1+tmp)
    else:
        e1 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23-r32)**2)/(3-tmp))

    tmp=-r11+r22-r33
    if tmp>0:
        e2 = 0.5*np.sqrt(1+tmp)
    else:
        e2 = 0.5*np.sqrt(((r12+r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    tmp=-r11+-22+r33
    if tmp>0:
        e3 = 0.5*np.sqrt(1+tmp)
    else:
        e3 = 0.5*np.sqrt(((r12-r21)**2+(r13+r31)**2+(r23+r32)**2)/(3-tmp))

    return np.array([[e0], [e1], [e2], [e3]])


def Euler2Quaternion(phi, theta, psi):
    """
    Converts an euler angle attitude to a quaternian attitude
    :param euler: Euler angle attitude in a np.matrix(phi, theta, psi)
    :return: Quaternian attitude in np.array(e0, e1, e2, e3)
    """

    e0 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
    e1 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
    e2 = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
    e3 = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

    return np.array([[e0],[e1],[e2],[e3]])

    
def Quaternion2Rotation(quaternion):
    """
    converts a quaternion attitude to a rotation matrix
    """
    e0 = quaternion.item(0)
    e1 = quaternion.item(1)
    e2 = quaternion.item(2)
    e3 = quaternion.item(3)

    R = np.array([[e1 ** 2.0 + e0 ** 2.0 - e2 ** 2.0 - e3 ** 2.0, 2.0 * (e1 * e2 - e3 * e0), 2.0 * (e1 * e3 + e2 * e0)],
                  [2.0 * (e1 * e2 + e3 * e0), e2 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e3 ** 2.0, 2.0 * (e2 * e3 - e1 * e0)],
                  [2.0 * (e1 * e3 - e2 * e0), 2.0 * (e2 * e3 + e1 * e0), e3 ** 2.0 + e0 ** 2.0 - e1 ** 2.0 - e2 ** 2.0]])
    R = R/linalg.det(R)

    return R