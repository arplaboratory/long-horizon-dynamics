import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from utils import check_folder_paths, plot_data
from config import parse_args
from data import DynamicsDataset
from models.mlp import MLP

import sys
import time 
import os
from tqdm import tqdm

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
    features = ['u', 'v', 'w',
                'phi', 'theta', 'psi', 
                'p', 'q', 'r', 
                'delta_e', 'delta_a', 'delta_r', 'delta_t']
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id) + "\n")

    # create the dataset
    train_dataset = DynamicsDataset(data_path + "train/", args.batch_size, features, args.normalize)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    valid_dataset = DynamicsDataset(data_path + "valid/", args.batch_size, features, args.normalize)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    if args.plot == True:
        plot_data(train_dataset.X, features = features, 
                                   save_path = experiment_path + "plots")
    print('Loading model ...')
    # Initialize the model, loss function, and optimizer
    model = MLP(input_size=train_dataset.X.shape[0], 
                output_size=train_dataset.Y.shape[0],
                num_layers=[512, 512, 256, 128, 64], 
                dropout=args.dropout).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss_list = []
    valid_loss_list = []

    print('Start training ...\n')
    try:
        best_loss = 1e8
        for epoch in range(1, args.epochs + 1):
            train_loss_sum = 0.0
            valid_loss_sum = 0.0

            batch = tqdm(train_dataloader, total=len(train_dataloader), desc="Train - Epoch {:2d} ".format(epoch))
            model.train()
            for i, (x, y) in enumerate(batch):

                for p in model.parameters():
                    p.grad = None
                
                x = x.to(args.device).float()
                y = y.to(args.device).float()

                y_pred = model(x)
                loss = criterion(y_pred, y)
                
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() 
                batch.set_postfix(loss=train_loss_sum / (i+1))
            
            # validate model
            if epoch % args.val_freq == 0:
                batch = tqdm(valid_dataloader, total=len(valid_dataloader), desc="Valid - Epoch {:2d} ".format(epoch))
                model.eval()
                for i, (x, y) in enumerate(batch):
                    x = x.to(args.device).float()
                    y = y.to(args.device).float()

                    with torch.no_grad():
                        y_pred = model(x)
                        loss = criterion(y_pred, y)
                        valid_loss_sum += loss.item()
                        batch.set_postfix(loss=valid_loss_sum / (i+1))

                if valid_loss_sum < best_loss:
                    best_loss = valid_loss_sum
                    torch.save(model.state_dict(), experiment_path + "checkpoints/model_epoch_" + str(epoch) + ".pth")
                    print('Saving model ...')
      
            train_loss_list.append(train_loss_sum / len(train_dataloader))
            valid_loss_list.append(valid_loss_sum / len(valid_dataloader))
    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early ...")
        print("-" * 89)
            
    # plotting loss curves
    train_loss_np = np.array(train_loss_list)
    valid_loss_np = np.array(valid_loss_list)

    epochs = list(range(1, len(train_loss_list) + 1))

    # plot loss curves
    plt.figure()
    plt.plot(epochs, train_loss_np, label="Train Loss")
    plt.plot(epochs, valid_loss_np, label="Valid Loss")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(experiment_path + "plots/loss.png")


            



    '''
    
    for epoch in range(args.epochs):
        train_loss_per_epoch = 0.0
        test_loss_per_epoch = 0.0

        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(args.device).float()
            y = y.to(args.device).float()
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()
        
        print('--------------------------------------------------------------------------------------')
        train_loss_list.append(np.mean(train_loss_per_epoch))

        # train_loss_list.append(train_loss_per_epoch * args.batch_size/len(train_dataset))
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {np.mean(train_loss_per_epoch)}")
        
        # validate model
        model.eval()
        for i, (x, y) in enumerate(test_dataloader):
            x = x.to(args.device).float()
            y = y.to(args.device).float()

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)
                test_loss_per_epoch += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {np.mean(test_loss_per_epoch)}")
        print('--------------------------------------------------------------------------------------')
        test_loss_list.append(np.mean(test_loss_per_epoch))

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), experiment_path + "checkpoints/model_epoch_" + str(epoch) + ".pth")

    # converting loss values to log scale
    train_loss_np = np.array(train_loss_list)
    test_loss_np = np.array(test_loss_list)

    epochs = list(range(1, len(train_loss_list) + 1))

    # plot loss curves
    plt.figure()
    plt.plot(epochs, train_loss_np, label="Train Loss")
    plt.plot(epochs, test_loss_np, label="Test Loss")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epochs")
    plt.ylabel("Logarihtmic Scale Loss")
    plt.savefig(experiment_path + "plots/loss.png")
    '''
    





