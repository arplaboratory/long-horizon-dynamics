import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval


class DynamicsDataset(Dataset):
    def __init__(self, data_path, batch_size, features, normalize):

        self.normalize = normalize
        self.mean = np.zeros((len(features), 1))
        self.std = np.ones((len(features), 1))
        raw_data = self.load_data(data_path)
        self.X, self.Y = self.load_input_and_output(raw_data, features)
        self.batch_size = batch_size
        self.num_steps = np.ceil(self.X.shape[1] / self.batch_size)

        assert self.X.shape[1] == self.Y.shape[1]

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[:, idx], self.Y[:, idx]
    
    def load_data(self, data_path):
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
    def load_input_and_output(self, raw_data, features):
        data_list = []
        for i, data in enumerate(raw_data):
            X = []
            for feature in features:
                X.append(data[feature])
            
            X = np.array(X)
            data_list.append(X)
        
        data_np = np.concatenate(data_list, axis=1)

        # # Remove wind from data
        # data_np = np.delete(data_np, [0, 1, 2, 11, 12], axis=0)

        # Normalization of data
        if self.normalize:
            data_np = (data_np - data_np.mean(axis=1).reshape(-1, 1)) / (data_np.std(axis=1).reshape(-1, 1))
            self.mean = data_np.mean(axis=1).reshape(-1, 1)
            self.std = data_np.std(axis=1).reshape(-1, 1)

        x = np.zeros((data_np.shape[0], data_np.shape[1]-1))
        y = np.zeros((data_np.shape[0] - 4, data_np.shape[1]-1))

        for i in range(data_np.shape[1] - 1):
            x[:, i] = data_np[:, i] 
            y[:8, i] = data_np[:8, i+1]

        return x, y
    
    