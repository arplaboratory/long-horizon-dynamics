import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from dynamics_learning.utils import Euler2Quaternion
from config import parse_args

def extract_data(data, dataset_name):
    try:
        if dataset_name == "pi_tcn":
            velocity_data = data[['v_x', 'v_y', 'v_z']].values
            attitude_data = data[['q_w', 'q_x', 'q_y', 'q_z']].values
            angular_velocity_data = data[['w_x', 'w_y', 'w_z']].values
            control_data = data[['u_0', 'u_1', 'u_2', 'u_3']].values * 0.001

        elif dataset_name == "neurobem":
            velocity_data = data[['vel x', 'vel y', 'vel z']].values
            attitude_data = data[['quat w', 'quat x', 'quat y', 'quat z']].values
            angular_velocity_data = data[['ang vel x', 'ang vel y', 'ang vel z']].values
            control_data = data[['mot 1', 'mot 2', 'mot 3', 'mot 4']].values * 0.001
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
    except KeyError as e:
        raise KeyError(f"Invalid field in dataset '{dataset_name}': {e}")

    return velocity_data, attitude_data, angular_velocity_data, control_data


def csv_to_hdf5(args, data_path):

    hdf5(data_path, 'train/', 'train.h5',  args.dataset,  args.history_length, args.unroll_length)
    hdf5(data_path, 'valid/', 'valid.h5',  args.dataset,  args.history_length, args.unroll_length)
    hdf5_trajectories(data_path, 'test/',  args.dataset,  args.history_length, 60)

def hdf5(data_path, folder_name, hdf5_file, dataset, history_length, unroll_length):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            # Modify time to start from 0
            data['t'] = data['t'] - data['t'].values[0]

            data['t'] = pd.to_datetime(data['t'], unit='s')

            data.set_index('t', inplace=True)
            data = data.resample('0.01s').mean()
            data.reset_index(inplace=True)

            velocity_data, attitude_data, angular_velocity_data, control_data = extract_data(data, dataset)
            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))

            num_samples = data_np.shape[0] - history_length - unroll_length
            if num_samples <= 0:
                print(f"Skipping file {file} due to insufficient data")
                continue

            X = np.zeros((num_samples, history_length, data_np.shape[1]))
            Y = np.zeros((num_samples, unroll_length, data_np.shape[1]))

            for i in range(num_samples):
                X[i, :, :] =   data_np[i:i+history_length, :]
                Y[i,:,:]   =   data_np[i+history_length:i+history_length+unroll_length,:data_np.shape[1]]

            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)    
        
    # save the data
    # Create the HDF5 file and datasets for inputs and outputs
    with h5py.File(data_path + folder_name + hdf5_file, 'w') as hf:
        inputs_data = hf.create_dataset('inputs', data=X)
        inputs_data.dims[0].label = 'num_samples'
        inputs_data.dims[1].label = 'history_length'
        inputs_data.dims[2].label = 'features'

        outputs_data = hf.create_dataset('outputs', data=Y)
        outputs_data.dims[0].label = 'num_samples'
        outputs_data.dims[1].label = 'unroll_length'
        outputs_data.dims[2].label = 'features'

        # flush and close the file
        hf.flush()
        hf.close()
        
    return X, Y

def hdf5_trajectories(data_path, folder_name, dataset, history_length, unroll_length):

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            # Modify time to start from 0
            data['t'] = data['t'] - data['t'].values[0]

            data['t'] = pd.to_datetime(data['t'], unit='s')

            data.set_index('t', inplace=True)
            data = data.resample('0.01s').mean()
            data.reset_index(inplace=True)

            velocity_data, attitude_data, angular_velocity_data, control_data = extract_data(data, dataset)

            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))
            num_samples = data_np.shape[0] - history_length - unroll_length
            if num_samples <= 0:
                print(f"Skipping file {file} due to insufficient data")
                continue

            X = np.zeros((num_samples, history_length, data_np.shape[1]))
            Y = np.zeros((num_samples, unroll_length, data_np.shape[1]))

            for i in range(num_samples):
                X[i, :, :] =   data_np[i:i+history_length, :]
                Y[i,:,:]   =   data_np[i+history_length:i+history_length+unroll_length,:data_np.shape[1]]

            # Save to hdf5 with the same name as the csv file
            with h5py.File(data_path + folder_name + file[:-4] + '.h5', 'w') as hf: 
                inputs_data = hf.create_dataset('inputs', data=X)
                inputs_data.dims[0].label = 'num_samples'
                inputs_data.dims[1].label = 'history_length'
                inputs_data.dims[2].label = 'features'

                outputs_data = hf.create_dataset('outputs', data=Y)
                outputs_data.dims[0].label = 'num_samples'
                outputs_data.dims[1].label = 'unroll_length'
                outputs_data.dims[2].label = 'features'

                # flush and close the file
                hf.flush()
                hf.close()    
                
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
    data_path = resources_path + "data/" + args.dataset + "/" 
    
    csv_to_hdf5(args, data_path)