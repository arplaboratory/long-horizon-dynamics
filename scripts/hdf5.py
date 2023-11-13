import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from utils import Euler2Quaternion, Euler2Rotation
from config import parse_args


def csv_to_hdf5(args, data_path):

    hdf5(data_path + 'train/', 'train.h5',  args.attitude,  args.history_length, args.unroll_length)
    hdf5(data_path + 'valid/', 'valid.h5',  args.attitude,  args.history_length, args.unroll_length)
    # hdf5(data_path + 'test/',  'test.h5',   args.attitude,  args.history_length, args.unroll_length)
    hdf5_test(data_path + 'test/',  'test.h5',   args.attitude,  args.history_length)


def hdf5(data_path, hdf5_file, attitude, history_length, unroll_length):

    all_X = []
    all_Y = []

    attitude_func = None    

    if attitude != 'euler':
        attitude_func = Euler2Quaternion if attitude == 'quaternion' else Euler2Rotation

    # load the data
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path, file)
            data = pd.read_csv(csv_file_path)


            if attitude == 'euler':
                attitude_data = data[['phi', 'theta', 'psi']].values
            
            elif attitude == 'quaternion':
                attitude_euler = data[['phi', 'theta', 'psi']].values
                attitude_data = np.zeros((attitude_euler.shape[0], 4))
                for i in range(attitude_euler.shape[0]):
                    attitude_data[i,:] = Euler2Quaternion(attitude_euler[i,0], attitude_euler[i,1], attitude_euler[i,2]).T
                
            elif attitude == 'rotation':
   
                attitude_euler = data[['phi', 'theta', 'psi']].values
                attitude_data = np.zeros((attitude_euler.shape[0], 9))
                for i in range(attitude_euler.shape[0]):
                    attitude_data[i,:] = Euler2Rotation(attitude_euler[i,0], attitude_euler[i,1], attitude_euler[i,2]).flatten()

            velocity_data = data[['u', 'v', 'w']].values
            angular_velocity_data = data[['p', 'q', 'r']].values
            control_data = data[['delta_e', 'delta_a', 'delta_r', 'delta_t']].values


            # Concatenate all data
            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))

            if history_length == 0:
                num_samples = data_np.shape[0] - 1

                X = data_np[:-1, :]
                Y = data_np[1:, :-4]
            else:
                num_samples = data_np.shape[0] - history_length - unroll_length
                X = np.lib.stride_tricks.as_strided(data_np, shape=(num_samples, history_length, data_np.shape[1]), strides=(data_np.strides[0], data_np.strides[0], data_np.strides[1]))
                Y = np.lib.stride_tricks.as_strided(data_np[history_length:], shape=(num_samples, unroll_length, data_np.shape[1]), strides=(data_np.strides[0], data_np.strides[0], data_np.strides[1]))

            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # if normalize save mean and std for each feature in the training set and normalize the data
    if args.normalize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        np.save(data_path + 'mean.npy', mean)
        np.save(data_path + 'std.npy', std)
        X = (X - mean) / std

    # save the data
    with h5py.File(data_path + hdf5_file, 'w') as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)

    return X, Y

def hdf5_test(data_path, hdf5_file, attitude, history_length):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path, file)

            data = pd.read_csv(csv_file_path)
            
            if attitude == 'euler':
                attitude_data = data[['phi', 'theta', 'psi']].values
            
            elif attitude == 'quaternion':
                attitude_euler = data[['phi', 'theta', 'psi']].values
                attitude_data = np.zeros((attitude_euler.shape[0], 4))
                for i in range(attitude_euler.shape[0]):
                    attitude_data[i,:] = Euler2Quaternion(attitude_euler[i,0], attitude_euler[i,1], attitude_euler[i,2]).T
                
            elif attitude == 'rotation':
   
                attitude_euler = data[['phi', 'theta', 'psi']].values
                attitude_data = np.zeros((attitude_euler.shape[0], 9))
                for i in range(attitude_euler.shape[0]):
                    attitude_data[i,:] = Euler2Rotation(attitude_euler[i,0], attitude_euler[i,1], attitude_euler[i,2]).flatten()

            position_data = data[['pn', 'pe', 'pd']].values
            velocity_data = data[['u', 'v', 'w']].values
            angular_velocity_data = data[['p', 'q', 'r']].values
            control_data = data[['delta_e', 'delta_a', 'delta_r', 'delta_t']].values

           
       
            data_np = np.hstack((velocity_data,
                                    attitude_data, angular_velocity_data, 
                                    control_data))
        
            
            num_samples = data_np.shape[0] - 1
            # Input features at the current time step
            X = np.zeros((num_samples, data_np.shape[1]))

            # Output rotation at the next time step excluding the control inputs
            Y = np.zeros((num_samples, data_np.shape[1]))
            for i in range(num_samples):
                X[i,:] = data_np[i,:]
                Y[i,:] = data_np[i+1, :data_np.shape[1]]


            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # if normalize save mean and std for each feature in the training set and normalize the data
    if args.normalize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        np.save(data_path + 'mean.npy', mean)
        np.save(data_path + 'std.npy', std)
        X = (X - mean) / std
    

    # save the data
    with h5py.File(data_path + hdf5_file, 'w') as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)

    return X, Y
                
# load hdf5
def load_hdf5(data_path, hdf5_file):
    with h5py.File(data_path + hdf5_file, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X.T, Y.T

if __name__ == "__main__":
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    
    csv_to_hdf5(args, data_path)

    # parse arguments
    

    X, Y = load_hdf5(data_path + 'test/', 'test.h5')
    print(X.shape, Y.shape)

    # print first row of data
    print(X[:, :, 0])
    print("-----------------------------")
    print(Y[:, :, 0])