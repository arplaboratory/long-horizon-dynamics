import os
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from dynamics_learning.utils import Quaternion2Euler, Quaternion2Rotation, deltaQuaternion
from config import parse_args


def quaternion_log(q):
    
        # Compute the log of a quaternion
        # Input: q = [q_w, q_x, q_y, q_z]
    
        # Compute the norm of the quaternion
    
        norm_q = np.linalg.norm(q, axis=1, keepdims=True) 
    
        # Get vector part of the quaternion
        q_v = q[:, 1:]
        q_v_norm = np.linalg.norm(q_v, axis=1, keepdims=True)

        # Compute the angle of rotation
        theta = 2 * np.arctan2(q_v_norm, q[:, 0:1])

        # Ompute the log of the quaternion
        q_log = theta * q_v / q_v_norm
    
        return q_log

def quaternion_difference(q_t1, q_t0):

    # Compute the rotation q that takes q_t0 to q_t1
    # Input: q_t1 = [q_w, q_x, q_y, q_z]
    #        q_t0 = [q_w, q_x, q_y, q_z]

    # Compute the norm of the quaternion
    norm_q_t1 = np.linalg.norm(q_t1, axis=1, keepdims=True)
    norm_q_t0 = np.linalg.norm(q_t0, axis=1, keepdims=True)

    # Normalize the quaternion
    q_t1 = q_t1 / norm_q_t1
    q_t0 = q_t0 / norm_q_t0

    # q_t0 inverse
    q_t0_inv = np.concatenate((q_t0[:, 0:1], -q_t0[:, 1:]), axis=1)

    # Compute the difference between the two quaternions
    q_diff = quaternion_product(q_t1, q_t0_inv)

    return q_diff

def quaternion_product(q_t1, q_t0):
    
        # Compute the rotation q that takes q_t0 to q_t1
        # Input: q_t1 = [q_w, q_x, q_y, q_z]
        #        q_t0 = [q_w, q_x, q_y, q_z]
    
        # Compute the norm of the quaternion
        norm_q_t1 = np.linalg.norm(q_t1, axis=1, keepdims=True)
        norm_q_t0 = np.linalg.norm(q_t0, axis=1, keepdims=True)
    
        # Normalize the quaternion
        q_t1 = q_t1 / norm_q_t1
        q_t0 = q_t0 / norm_q_t0
    
        # Compute the quaternion product
        q_hat = np.concatenate((q_t1[:, 0:1] * q_t0[:, 0:1] - q_t1[:, 1:2] * q_t0[:, 1:2] - q_t1[:, 2:3] * q_t0[:, 2:3] - q_t1[:, 3:4] * q_t0[:, 3:4],
                                q_t1[:, 0:1] * q_t0[:, 1:2] + q_t1[:, 1:2] * q_t0[:, 0:1] + q_t1[:, 2:3] * q_t0[:, 3:4] - q_t1[:, 3:4] * q_t0[:, 2:3],
                                q_t1[:, 0:1] * q_t0[:, 2:3] - q_t1[:, 1:2] * q_t0[:, 3:4] + q_t1[:, 2:3] * q_t0[:, 0:1] + q_t1[:, 3:4] * q_t0[:, 1:2],
                                q_t1[:, 0:1] * q_t0[:, 3:4] + q_t1[:, 1:2] * q_t0[:, 2:3] - q_t1[:, 2:3] * q_t0[:, 1:2] + q_t1[:, 3:4] * q_t0[:, 0:1]), axis=1)
    
        return q_hat


def csv_to_hdf5(args, data_path):

    hdf5(data_path, 'train/', 'train.h5',  args.attitude,  args.history_length, args.unroll_length)
    hdf5(data_path, 'valid/', 'valid.h5',  args.attitude,  args.history_length, args.unroll_length, train=False)
    hdf5(data_path, 'test/',  'test.h5',   args.attitude,  args.history_length, args.unroll_length, train=False)
    # hdf5_recursive(data_path, 'test/',  'test_eval.h5')

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

            num_samples = data_np.shape[0] - history_length - unroll_length

            X = np.zeros((num_samples, history_length, data_np.shape[1]))
            Y = np.zeros((num_samples, unroll_length, data_np.shape[1]))

            for i in range(num_samples):
                X[i, :, :] =   data_np[i:i+history_length, :]
                Y[i,:,:]   =   data_np[i+history_length: i+history_length+unroll_length,:data_np.shape[1]]

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
    
    # csv_to_hdf5(args, data_path)


    X, Y = load_hdf5(data_path + 'train/', 'train.h5')
    
    # q_0 = np.array([[0.9848077, -0.1736483, 0, 0]])
    # q_1 = np.array([[1, 0, 0, 0]])

    # print(q_0.shape, q_1.shape)

    # q_diff = quaternion_difference(q_1, q_0)
    # q_diff_log = quaternion_log(q_diff)

    # print(q_diff_log)



    ############## Data Analysis ##############

    # Absolute difference between the last state of the input and the output
    # print("Absolute difference between the last state of the input and the output")
    # print(np.mean(np.abs(X[:, -1, :-4] - Y[:, 0, :-4]), axis=0))

    # # Varience 
    # print(np.var(np.abs(X[:, -1, :-4] - Y[:, 0, :-4]), axis=0))


    # Get quaternion between the last state of the input and the output
    # print("Quaternion between the last state of the input and the output")

    quat_diff = []
    for i in range(X.shape[0]):
        q_t0 = X[i, -1, 3:7]
        q_t1 = Y[i, 0, 3:7]

        # Expanding the dimensions
        q_t0 = np.expand_dims(q_t0, axis=0)
        q_t1 = np.expand_dims(q_t1, axis=0)
        
        quat_error = quaternion_difference(q_t1, q_t0)
        quat_error_log = quaternion_log(quat_error)

        quat_diff.append(quat_error_log)
    
    abs_diff = np.array(quat_diff)
    print(np.mean(abs_diff, axis=0))
    print(np.var(abs_diff, axis=0))



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