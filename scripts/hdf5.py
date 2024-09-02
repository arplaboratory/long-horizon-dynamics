import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from dynamics_learning.utils import Euler2Quaternion
from config import parse_args

def extract_data(data, dataset_name):
   

    velocity_data = data[['vx', 'vy', 'vz']].values
    attitude_data = data[['q0', 'q1', 'q2', 'q3']].values
    angular_velocity_data = data[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values
    airspeed = data['airspeed'].values
    wind = data[['wind_north', 'wind_east']].values
    differential_pressure = data['diff_pressure'].values
    control_data = data[['throttle', 'aileron', 'elevator', 'rudder']].values

    return velocity_data, attitude_data, angular_velocity_data, airspeed, wind, differential_pressure, control_data


def csv_to_hdf5(args, data_path):

    print("Converting csv files to hdf5 ...", data_path)

    hdf5(data_path, 'train/', 'train.h5',  args.dataset,  args.history_length, args.unroll_length, args.augment_input)
    hdf5(data_path, 'valid/', 'valid.h5',  args.dataset,  args.history_length, args.unroll_length, args.augment_input)
    hdf5_trajectories(data_path, 'test/',  args.dataset,  args.history_length, 60, args.augment_input)

def hdf5(data_path, folder_name, hdf5_file, dataset, history_length, unroll_length, augment_input):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data, attitude_data, angular_velocity_data, airspeed, wind, differential_pressure, control_data = extract_data(data, dataset)

            if augment_input == 'va':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, airspeed.reshape(-1, 1), control_data))
            elif augment_input == 'w':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, wind, control_data))
            elif augment_input == 'P':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, differential_pressure.reshape(-1, 1), control_data))
            elif augment_input == 'vawP':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, airspeed.reshape(-1, 1), wind, differential_pressure.reshape(-1, 1), control_data))
                
            # data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))

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

def hdf5_trajectories(data_path, folder_name, dataset, history_length, unroll_length, augment_input):

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data, attitude_data, angular_velocity_data, airspeed, wind, differential_pressure, control_data = extract_data(data, dataset)

            if augment_input == 'va':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, airspeed.reshape(-1, 1), control_data))
            elif augment_input == 'w':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, wind, control_data))
            elif augment_input == 'P':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, differential_pressure.reshape(-1, 1), control_data))
            elif augment_input == 'vawP':
                data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, airspeed.reshape(-1, 1), wind, differential_pressure.reshape(-1, 1), control_data))
                
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