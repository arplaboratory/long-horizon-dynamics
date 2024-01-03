import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from dynamics_learning.utils import Quaternion2Euler, Quaternion2Rotation
from config import parse_args


def csv_to_hdf5(args, data_path):

    hdf5(data_path, 'train/', 'train.h5',  args.attitude,  args.history_length, args.unroll_length)
    hdf5(data_path, 'valid/', 'valid.h5',  args.attitude,  args.history_length, args.unroll_length, train=False)
    hdf5(data_path, 'test/',  'test.h5',   args.attitude,  args.history_length, args.unroll_length, train=False)
    hdf5_recursive(data_path, 'test/',  'test_eval.h5')

def hdf5(data_path, folder_name, hdf5_file, attitude, history_length, unroll_length, train=True):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data = data[['v_x', 'v_y', 'v_z']].values
            attitude_data = data[['q_w', 'q_x', 'q_y', 'q_z']].values
            angular_velocity_data = data[['w_x', 'w_y', 'w_z']].values
            control_data = data[['u_0', 'u_1', 'u_2', 'u_3']].values * 0.001 

            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))

            num_samples = data_np.shape[0] - history_length 

            X = np.zeros((num_samples, history_length, data_np.shape[1]))
            Y = np.zeros((num_samples, data_np.shape[1]-4))

            for i in range(num_samples):
                X[i, :, :] = data_np[i:i+history_length, :]
                Y[i, :] = data_np[i+history_length, :-4]

            # If delta is true, then the output is the difference between the current and previous state
            if args.delta:
                Y = Y - X[:, -1, :-4]
            
            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # Normalize the output 
    if args.normalize:
        if train == True:
            mean = np.mean(Y, axis=0)
            std = np.std(Y, axis=0)

            np.save(data_path + folder_name + 'mean_train.npy', mean)
            np.save(data_path + folder_name + 'std_train.npy', std)
        else:
            mean = np.load(data_path + 'train/' + 'mean_train.npy')
            std = np.load(data_path + 'train/' + 'std_train.npy')

        Y = (Y - mean) / std
    
        
    # save the data
    # Create the HDF5 file and datasets for inputs and outputs
    with h5py.File(data_path + folder_name + hdf5_file, 'w') as hf:
        inputs_data = hf.create_dataset('inputs', data=X)
        inputs_data.dims[0].label = 'num_samples'
        inputs_data.dims[1].label = 'history_length'
        inputs_data.dims[2].label = 'features'

        outputs_data = hf.create_dataset('outputs', data=Y)
        outputs_data.dims[0].label = 'num_samples'
        outputs_data.dims[1].label = 'features'

        # flush and close the file
        hf.flush()
        hf.close()
        
    return X, Y

def hdf5_recursive(data_path, folder_name, hdf5_file):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data = data[['v_x', 'v_y', 'v_z']].values
            attitude_data = data[['q_w', 'q_x', 'q_y', 'q_z']].values
            angular_velocity_data = data[['w_x', 'w_y', 'w_z']].values
            control_data = data[['u_0', 'u_1', 'u_2', 'u_3']].values * 0.001 

            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))

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

    
        
    # save the data
    # Create the HDF5 file and datasets for inputs and outputs
    with h5py.File(data_path + folder_name + hdf5_file, 'w') as hf:
        inputs_data = hf.create_dataset('inputs', data=X)
        inputs_data.dims[0].label = 'num_samples'
        inputs_data.dims[1].label = 'features'

        outputs_data = hf.create_dataset('outputs', data=Y)
        outputs_data.dims[0].label = 'num_samples'
        outputs_data.dims[1].label = 'features'

        # flush and close the file
        hf.flush()
        hf.close()
        
    return X, Y

                
# load hdf5
def load_hdf5(data_path, hdf5_file):
    with h5py.File(data_path + hdf5_file, 'r') as hf:
        X = hf['inputs'][:]
        Y = hf['outputs'][:]

    return X, Y

if __name__ == "__main__":
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/" + "quadrotor/"
    
    csv_to_hdf5(args, data_path)


    X, Y = load_hdf5(data_path + 'test/', 'test_eval.h5')
    print(X.shape, Y.shape)


    # print("Min and Max values for each of the output features")
    # print("Minimum")
    # print(np.min(Y, axis=0))

    # print("Maximum")
    # print(np.max(Y, axis=0))


    # # Print the MSE between the last state of the input and the output
    # print("MSE")
    # print(np.mean(np.square(X[:, -1, :-4] - Y), axis=0))


    # X, Y = load_hdf5(data_path + 'test/', 'test_eval.h5')
    # print(X.shape, Y.shape)

    # print(Y[:, 0, -2:])

    # X, Y = load_hdf5(data_path + 'test/', 'test_trajectory.h5')
    # print(X.shape, Y.shape)





   


    # X, Y = load_hdf5(data_path + 'test/', 'test_eval.h5')
    # print(X.shape, Y.shape)

    # print(Y[:, 0, -2:])

    # X, Y = load_hdf5(data_path + 'test/', 'test_trajectory.h5')
    # print(X.shape, Y.shape)