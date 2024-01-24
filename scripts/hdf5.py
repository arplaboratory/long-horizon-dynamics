import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from dynamics_learning.utils import Euler2Quaternion, quaternion_log, quaternion_difference, quaternion_product
from config import parse_args

SAMPLING_FREQUENCY = {'fixed_wing': 100, 'quadrotor': 100, 'neurobem': 400}



def extract_data(data, dataset_name):
    try:
        if dataset_name == "fixed_wing":
            velocity_data = data[['u', 'v', 'w']].values
            euler_data = data[['phi', 'theta', 'psi']].values
            
            attitude_data = np.zeros((euler_data.shape[0], 4))
            for i in range(euler_data.shape[0]):
                attitude_data[i, :] = Euler2Quaternion(euler_data[i,0], euler_data[i,1], euler_data[i,2]).T

            angular_velocity_data = data[['p', 'q', 'r']].values
            control_data = data[['delta_e', 'delta_a', 'delta_r', 'delta_t']]

        elif dataset_name == "quadrotor":
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

    hdf5(data_path, 'train/', 'train.h5',  args.vehicle_type,  args.attitude,  args.history_length, args.unroll_length, args.sampling_frequency)
    hdf5(data_path, 'valid/', 'valid.h5',  args.vehicle_type,  args.attitude,  args.history_length, args.unroll_length, args.sampling_frequency)
    hdf5(data_path, 'test/',  'test.h5',   args.vehicle_type,  args.attitude,  args.history_length, args.unroll_length, args.sampling_frequency)
    hdf5_recursive(data_path, 'test/',  'test_eval.h5', args.vehicle_type)

def hdf5(data_path, folder_name, hdf5_file, dataset, attitude, history_length, unroll_length, sampling_frequency):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data, attitude_data, angular_velocity_data, control_data = extract_data(data, dataset)

            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))

            # Sampling frequency
            data_np = data_np[::int(SAMPLING_FREQUENCY[dataset]/sampling_frequency), :]

            num_samples = data_np.shape[0] - history_length - unroll_length

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

def hdf5_recursive(data_path, folder_name, hdf5_file, dataset):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data, attitude_data, angular_velocity_data, control_data = extract_data(data, dataset)

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
    data_path = resources_path + "data/" + args.vehicle_type + "/" 
    
    csv_to_hdf5(args, data_path)


    X, Y = load_hdf5(data_path + 'train/', 'train.h5')
    
    print("Shape of the input data: ",  X.shape)
    print("Shape of the output data: ", Y.shape)


    ############## Data Analysis ##############

    # Absolute difference between the last state of the input and the output
    print("Absolute difference between the last state of the input and the output")
    print(np.mean(np.abs(X[:, -1, :-4] - Y[:, 0, :-4]), axis=0))

    # Varience 
    print(np.var(np.abs(X[:, -1, :-4] - Y[:, 0, :-4]), axis=0))

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