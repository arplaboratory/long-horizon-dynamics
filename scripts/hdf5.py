import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from utils import Euler2Quaternion, Euler2Rotation
from config import parse_args


def csv_to_hdf5(args, data_path):

    hdf5(data_path, 'train/', 'train.h5',  args.attitude,  args.history_length, args.unroll_length)
    hdf5(data_path, 'valid/', 'valid.h5',  args.attitude,  args.history_length, args.unroll_length, train=False)
    hdf5(data_path, 'test/',  'test_eval.h5',   args.attitude,  args.history_length, args.unroll_length, train=False)
    hdf5_test(data_path, 'test/',  'test_trajectory.h5',   args.attitude,  args.history_length)


def hdf5(data_path, folder_name, hdf5_file, attitude, history_length, unroll_length, train=True):

    all_X = []
    all_Y = []

    attitude_func = None    

    if attitude != 'euler':
        attitude_func = Euler2Quaternion if attitude == 'quaternion' else Euler2Rotation

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
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
                num_samples = data_np.shape[0] - history_length - unroll_length + 1
                X = np.zeros((num_samples, history_length, data_np.shape[1]))
                Y = np.zeros((num_samples, unroll_length, data_np.shape[1]))

                for i in range(num_samples):
                    X[i,:,:] = data_np[i:i+history_length,:]
                    Y[i,:,:] = data_np[i+history_length: i+history_length+unroll_length,:data_np.shape[1]]

                # output should be the difference between the unrolled states and the current state
                if args.delta:
                    # Y[:, :, :-4] = Y[:, :, :-4] - X[:, -1, :].reshape((num_samples, 1, data_np.shape[1])).repeat(unroll_length, axis=1)[:, :, :-4]
                    
                    # Difference in velocity
                    Y[:, :, :3] = Y[:, :, :3] - X[:, -1, :].reshape((num_samples, 1, data_np.shape[1])).repeat(unroll_length, axis=1)[:, :, :3]

                    # Difference in anglular velocity
                    Y[:, :, 12:15] = Y[:, :, 12:15] - X[:, -1, :].reshape((num_samples, 1, data_np.shape[1])).repeat(unroll_length, axis=1)[:, :, 12:15]
                    

               
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
        outputs_data.dims[1].label = 'unroll_length'
        outputs_data.dims[2].label = 'features'

        # flush and close the file
        hf.flush()
        hf.close()
        
    return X, Y

def hdf5_test(data_path, folder_name, hdf5_file, attitude, history_length):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)

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

   
    # save the data
    # Create the HDF5 file and datasets for inputs and outputs
    with h5py.File(data_path + folder_name + hdf5_file, 'w') as hf:
        inputs_data = hf.create_dataset('inputs', data=X)
        inputs_data.dims[0].label = 'num_samples'
        # inputs_data.dims[1].label = 'history_length'
        inputs_data.dims[1].label = 'features'

        outputs_data = hf.create_dataset('outputs', data=Y)
        outputs_data.dims[0].label = 'num_samples'
        # outputs_data.dims[1].label = 'unroll_length'
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

    return X.T, Y.T

if __name__ == "__main__":
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    
    csv_to_hdf5(args, data_path)

    # parse arguments
    

    X, Y = load_hdf5(data_path + 'test/', 'test_eval.h5')
    print(X.shape, Y.shape)

    print(Y[:, 0, -2:])

    X, Y = load_hdf5(data_path + 'test/', 'test_trajectory.h5')
    print(X.shape, Y.shape)