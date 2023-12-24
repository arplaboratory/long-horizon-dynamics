import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from dynamics_learning.utils import Euler2Quaternion, Euler2Rotation
from config import parse_args


def csv_to_hdf5(args, data_path):

    hdf5(data_path, 'train/', 'train.h5',  args.attitude,  args.history_length, args.unroll_length)
    hdf5(data_path, 'valid/', 'valid.h5',  args.attitude,  args.history_length, args.unroll_length, train=False)
    hdf5(data_path, 'test/',  'test.h5',   args.attitude,  args.history_length, args.unroll_length, train=False)
    # hdf5_test(data_path, 'test/',  'test_trajectory.h5',   args.attitude,  args.history_length)


def hdf5(data_path, folder_name, hdf5_file, attitude, history_length, unroll_length, train=True):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            velocity_data = data[['v_x', 'v_y', 'v_z']].values
            attitude_data = data[['q_x', 'q_y', 'q_z', 'q_w']].values
            angular_velocity_data = data[['w_x', 'w_y', 'w_z']].values
            control_data = data[['u_0', 'u_1', 'u_2', 'u_3']].values * 0.001

            v_dot_data = data[['vdot_x', 'vdot_y', 'vdot_z']].values
            w_dot_data = data[['wdot_x', 'wdot_y', 'wdot_z']].values

            # v_dot_nom = data[['vdot_nom_x', 'vdot_nom_y', 'vdot_nom_z']].values
            # w_dot_nom = data[['wdot_nom_x', 'wdot_nom_y', 'wdot_nom_z']].values


            # Concatenate all data
            input_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data))
            # output_np = np.hstack((v_dot_data, w_dot_data, v_dot_nom, w_dot_nom))
            output_np = np.hstack((v_dot_data, w_dot_data))
            assert input_np.shape[0] == output_np.shape[0]
            

            if history_length == 0:
                num_samples = input_np.shape[0] - 1

                X = input_np
                Y = output_np
            else:
                num_samples = input_np.shape[0] - history_length 
                X = np.zeros((num_samples, history_length, input_np.shape[1]))
                Y = np.zeros((num_samples, output_np.shape[1]))

                for i in range(num_samples):
                    X[i,:,:] = input_np[i:i+history_length,:]
                    Y[i,:] = output_np[i+history_length-1,:]
        
               
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

    # parse arguments
    

    X, Y = load_hdf5(data_path + 'train/', 'train.h5')
    print(X.shape, Y.shape)

   
    print(X[:, :, 0])
    print(Y[:, 0])

    # print("Min and Max values for each of the output features")
    print("Minimum")
    print(np.min(Y, axis=1))

    print("Maximum")
    print(np.max(Y, axis=1))



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