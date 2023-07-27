import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils


class DynamicsDataset(Dataset):
    def __init__(self, data_path, batch_size):
        raw_data = utils.load_data(data_path)
        self.X, self.Y = utils.load_input_and_output(raw_data)
        self.batch_size = batch_size

        assert self.X.shape[1] == self.Y.shape[1]

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        return self.X[:, idx], self.Y[:, idx]
    