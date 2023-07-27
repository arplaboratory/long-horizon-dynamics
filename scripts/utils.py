import os
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


features = ['pn', 'pe', 'pd', 
            'phi', 'theta', 'psi', 
            'Va', 'Vg', 
            'p', 'q', 'r', 
            'wn', 'we', 
            'delta_e', 'delta_a', 'delta_r', 'delta_t']


def load_data(data_path):
    """Load csv from data_path folder

    Args:
        data_path (str): path to data folder

    Returns:
        ndarray
    """
    
    # Load data

    data = []
    filename = sorted(os.listdir(data_path))
    for file in filename:
        file_path = os.path.join(data_path, file)
        if not file_path.endswith('.csv'):
            continue
        df = pd.read_csv(file_path)

        
        # Convert string to ndarray
        for attributes in df.columns[1:]:
            if isinstance(df[attributes][0], str):
                df[attributes] = df[attributes].apply(literal_eval) 
        
        data.append({})
        for field in df.columns[1:]:
            data[-1][field] = np.array(df[field].tolist(), dtype=float)

    return data

def load_input_and_output(raw_data):
    """Load input and output from data

    Args:
        data (ndarray): data

    Returns:
        ndarray, ndarray
    """
    data_list = []
    for i, data in enumerate(raw_data):
        X = []
        for feature in features:
            X.append(data[feature])
        
        X = np.array(X)
        data_list.append(X)
    
    data_np = np.concatenate(data_list, axis=1)

    x = np.zeros((data_np.shape[0], data_np.shape[1]-1))
    y = np.zeros((data_np.shape[0], data_np.shape[1]-1))

    for i in range(data_np.shape[1] - 1):
        x[:, i] = data_np[:, i] 
        y[:, i] = data_np[:, i+1]

    return x, y

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

if __name__=="__main__":
    features = ['pn', 'pe', 'pd', 
                'phi', 'theta', 'psi', 
                'Va', 'Vg', 
                'p', 'q', 'r', 
                'wn', 'we', 
                'delta_e', 'delta_a', 'delta_r', 'delta_t']

    raw_data = load_data('/home/prat/arpl/TII/ws_dynamics/resources/data/train')
    x, y = load_input_and_output(raw_data)
    print(x.shape, y.shape)
    # plot_data(x, features, '/home/prat/arpl/TII/ws_dynamics/data/train')


