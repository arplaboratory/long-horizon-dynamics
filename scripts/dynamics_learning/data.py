import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

import h5py

class DynamicsDataset(Dataset):
    def __init__(self, data_path, hdf5_file, args):

        self.normalize = args.normalize
        self.X, self.Y = self.load_data(data_path, hdf5_file)        
        self.history_length = args.history_length
        self.unroll_length = args.unroll_length
        self.batch_size = args.batch_size

        self.num_steps = np.ceil(self.X.shape[0] / self.batch_size).astype(int)
        self.augmentations = args.augmentation
        self.std_percentage = args.std_percentage
        self.attitude = args.attitude
        self.model_type = args.model_type
        

        assert self.X.shape[0] == self.Y.shape[0]
        self.X_shape = self.X.shape
        self.Y_shape = self.Y.shape
        
        self.data_len = self.X_shape[0]
        self.state_len = self.X_shape[2]

    def __len__(self):

        return self.X_shape[0]

    def __getitem__(self, idx):
           

        x = self.X[idx, :, :]
        y = self.Y[idx, :, :]
            
        return x, y
        
    def load_data(self, hdf5_path, hdf5_file):
        with h5py.File(os.path.join(hdf5_path, hdf5_file), 'r') as hf: 
            X = hf['inputs'][:]
            Y = hf['outputs'][:]
        return X, Y
    
    
def load_dataset(mode, data_path, hdf5_file, args, num_workers, pin_memory):
    print('Generating', mode, 'data ...')
    dataset = DynamicsDataset(data_path, hdf5_file, args)
    batch_size = args.batch_size #if args.batch_size > 0 else len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=args.shuffle, num_workers=num_workers, pin_memory=pin_memory)
    print('... Loaded', dataset.data_len, 'points')
    print('|State|   =', dataset.state_len)
    print('|History| =', dataset.history_length)

    return dataset, loader