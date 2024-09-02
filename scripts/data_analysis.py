import os
import pandas as pd
import numpy as np
import h5py
import sys
from config import parse_args

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
    test_data_path = resources_path + "data/" + args.dataset + "/" + "test/"
    
    # Load test hdf5 files
    hdf5_files = [f for f in os.listdir(test_data_path) if f.endswith('.h5')]

    # Load the first hdf5 file
    X, Y = load_hdf5(test_data_path, hdf5_files[2])
    
    # Compute the mean abs difference between the the last input and the first output for each feature 
    mean_abs_diff = np.mean(np.abs(X[:,-1,:-4] - Y[:,0,:-4]), axis=0)

    print("Mean absolute difference between the last input and the first output for each feature:", mean_abs_diff)

    
