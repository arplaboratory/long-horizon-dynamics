import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import check_folder_paths, plot_data
from config import parse_args
from data import DynamicsDataset
from models.mlp import MLP

import sys
import time 
import os

if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    experiment_path = resources_path + "experiments/" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(args.run_id) + "/"

    check_folder_paths([experiment_path + "checkpoints",
                        experiment_path + "plots"])
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id))

    # create the dataset
    train_dataset = DynamicsDataset(data_path + "train/", args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    test_dataset = DynamicsDataset(data_path + "test/", args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    if args.plot == True:
        plot_data(train_dataset.X, features = ['pn', 'pe', 'pd', 
                                               'phi', 'theta', 'psi', 
                                               'Va', 'Vg', 
                                               'p', 'q', 'r', 
                                               'wn', 'we', 
                                               'delta_e', 'delta_a', 'delta_r', 'delta_t'], 
                                   save_path = experiment_path + "plots")
    

    # Initialize the model, loss function, and optimizer
    model = MLP(input_size=train_dataset.X.shape[0], 
                num_layers=[64, 64, 64], 
                dropout=args.dropout).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):
        train_loss_per_epoch = 0.0
        test_loss_per_epoch = 0.0

        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(args.device).float()
            y = y.to(args.device).float()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()
        
        print('--------------------------------------------------------------------------------------')
        train_loss_list.append(train_loss_per_epoch/len(train_dataset))
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss_per_epoch/len(train_dataset)}")
        
        # validate model
        model.eval()
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(args.device).float()
            y = y.to(args.device).float()

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                test_loss_per_epoch += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {test_loss_per_epoch/len(test_dataset)}")
        print('--------------------------------------------------------------------------------------')
        test_loss_list.append(test_loss_per_epoch/len(test_dataset))




