import os
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch

INPUT_FEATURES = ['u', 'v', 'w',
                      'r11', 'r21', 'r31', 
                      'r12', 'r22', 'r32',
                      'p', 'q', 'r',
                      'delta_e', 'delta_a', 'delta_r', 'delta_t']
OUTPUT_FEATURES = ['u', 'v', 'w',
                    'p', 'q', 'r',]

def load_data(data_path, input_features, output_features, use_history=False, history_length=4):
    """
    Read data from multiple CSV files in a folder and prepare concatenated input-output pairs.

    Args:
    - data_path (str): Path to the folder containing CSV files.
    - input_features (list): List of feature names to include in the input array.
    - output_features (list): List of feature names to include in the output array.
    - use_history (bool): If True, include historical state-action pairs in the input; if False, use only the current state-action pair.
    - history_length (int): Number of historical state-action pairs to consider if use_history is True.

    Returns:
    - X (numpy.ndarray): Concatenated input array with selected features.
    - Y (numpy.ndarray): Concatenated output array with selected features.
    """

    all_X = []
    all_Y = []

    for filename in os.listdir(data_path):
        if filename.endswith(".csv"):
            csv_file_path = os.path.join(data_path, filename)
            
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                data = [row for row in reader]

            num_samples = len(data) - (history_length if use_history else 1)
            num_input_features = len(input_features)
            num_output_features = len(output_features)

            if use_history:
                X = np.zeros((num_samples, history_length, num_input_features))
            else:
                X = np.zeros((num_samples, num_input_features))
                
            Y = np.zeros((num_samples, num_output_features))

            for i in range(num_samples):
                if use_history:
                    for j in range(history_length):
                        for k in range(num_input_features):
                            X[i, j, k] = float(data[i + j][input_features[k]])
                else:
                    for k in range(num_input_features):
                        X[i, k] = float(data[i][input_features[k]])

                for k in range(num_output_features):
                    Y[i, k] = float(data[i + (history_length if use_history else 1)][output_features[k]])

            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    return X, Y

def plot_data(data, features, save_path):
    """Plot data

    Args:
        data (ndarray): data
        features (List[str]): list of features
        save_path (str, optional): path to save figure. Defaults to None.
    """
    for i, feature in enumerate(features):
        plt.figure(i)
        plt.plot(data[i, :])
        plt.title(feature)
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel(feature)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, feature+'.png'))


def check_folder_paths(folder_paths):
    for path in folder_paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Creating folder", path, "...")

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

def Euler2Rotation(phi, theta, psi):
    """
    Converts euler angles to rotation matrix (R_b^i)
    """
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_roll = np.array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])
    R_pitch = np.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
    R_yaw = np.array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])
    #R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    R = R_yaw @ R_pitch @ R_roll

    # rotation is body to inertial frame
    # R = np.array([[c_theta*c_psi, s_phi*s_theta*c_psi-c_phi*s_psi, c_phi*s_theta*c_psi+s_phi*s_psi],
    #               [c_theta*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi, c_phi*s_theta*s_psi-s_phi*c_psi],
    #               [-s_theta, s_phi*c_theta, c_phi*c_theta]])

    return R

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
    R = R/np.linalg.det(R)

    return R

def deltaQuaternion(q1, q2):

    assert q1.shape == q2.shape

    q1_norms_sq = np.sum(q1**2, axis=1)

    q1_inv = np.hstack((q1[:, 0:1], -q1[:, 1:4])) / q1_norms_sq.reshape(-1, 1)

    # FInd the difference between the two quaternions
    delta_q = q2 * q1_inv

    return delta_q

def quaternion_product(q1, q2):
    
        # Compute the product of two quaternions
        # Input: q1 = [q_w, q_x, q_y, q_z]
        #        q2 = [q_w, q_x, q_y, q_z]
    
        w1, x1, y1, z1 = q1[:, 0:1], q1[:, 1:2], q1[:, 2:3], q1[:, 3:]
        w2, x2, y2, z2 = q2[:, 0:1], q2[:, 1:2], q2[:, 2:3], q2[:, 3:]

        # Compute the product of the two quaternions
        q_prod = torch.cat((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), dim=1)
        
        return q_prod

def quaternion_difference(q_pred, q_gt):

    # Compute the difference between two quaternions
    # Input: q_pred = [q_w, q_x, q_y, q_z]
    #        q_gt   = [q_w, q_x, q_y, q_z]

    # Compute the norm of the quaternion
    norm_q_pred = torch.norm(q_pred, dim=1, keepdim=True) 
    norm_q_gt = torch.norm(q_gt, dim=1, keepdim=True) 

    # Normalize the quaternion
    q_pred = q_pred / norm_q_pred
    q_gt = q_gt / norm_q_gt

    # q_pred inverse
    q_pred_inv = torch.cat((q_pred[:, 0:1], -q_pred[:, 1:]), dim=1)

    # Compute the difference between the two quaternions
    q_diff = quaternion_product(q_gt, q_pred_inv)

    return q_diff

def quaternion_log(q):
        
        # Compute the log of a quaternion
        # Input: q = [q_w, q_x, q_y, q_z]
        
        # Compute the norm of the quaternion
        
        # norm_q = torch.norm(q, dim=1, keepdim=True) 
        
        # Get vector part of the quaternion
        q_v = q[:, 1:]
        q_v_norm = torch.norm(q_v, dim=1, keepdim=True)
        
        # Compute the angle of rotation
        theta = 2 * torch.atan2(q_v_norm, q[:, 0:1])
        
        # Compute the log of the quaternion
        q_log = theta * q_v / q_v_norm
        
        return q_log

def quaternion_error(q_pred, q_gt):

    # Compute dot product between two quaternions
    # Input: q_pred = [q_w, q_x, q_y, q_z]
    #        q_gt   = [q_w, q_x, q_y, q_z]

    # Compute the norm of the quaternion
    norm_q_pred = torch.norm(q_pred, dim=1, keepdim=True)
    norm_q_gt = torch.norm(q_gt, dim=1, keepdim=True)

    # Normalize the quaternion
    q_pred = q_pred / norm_q_pred
    q_gt = q_gt / norm_q_gt

    # Compute the dot product between the two quaternions 
    q_dot = torch.sum(q_pred * q_gt, dim=1, keepdim=True)

    # Compute the angle between the two quaternions
    theta = torch.acos(torch.abs(q_dot))

    # min_theta = torch.min(theta, np.pi - theta)

    # COnvert to degrees
    # theta = theta * 180 / np.pi

    return theta


if __name__=="__main__":


    x, y = load_data('/home/prat/arpl/TII/ws_dynamics/FW-DYNAMICS_LEARNING/resources/data/train', INPUT_FEATURES, OUTPUT_FEATURES, history_length=4, use_history=True)
    print(x.shape, y.shape)

    print(x[0, :])
    print(y[0, :])

    # print(data)
    # print(x.shape, y.shape)
    # plot_data(x, features, '/home/prat/arpl/TII/ws_dynamics/data/train')

