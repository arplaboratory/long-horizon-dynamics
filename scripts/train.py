from torch.utils.data import DataLoader

from utils import check_folder_paths, plot_data
from config import parse_args, save_args
from data import DynamicsDataset
import pytorch_lightning
from lighting import DynamicsLearning


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
    
    # save arguments
    save_args(args, experiment_path + "args.txt")

    # set input and output features based on attitude type from args
    if args.attitude == "quaternion":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'e0', 'e1', 'e2', 'e3',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'e0', 'e1', 'e2', 'e3', 
                           'p', 'q', 'r']
    elif args.attitude == "rotation":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'r11', 'r12', 'r13', 
                          'r21', 'r22', 'r23',
                          'r31', 'r32', 'r33',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'r11', 'r12', 'r13', 
                           'r21', 'r22', 'r23',
                           'r31', 'r32', 'r33',
                           'p', 'q', 'r']
    elif args.attitude == "euler":
        INPUT_FEATURES = ['u', 'v', 'w',
                          'phi', 'theta', 'psi',
                          'p', 'q', 'r',
                          'delta_e', 'delta_a', 'delta_r', 'delta_t']
        OUTPUT_FEATURES = ['u', 'v', 'w',
                           'phi', 'theta', 'psi',
                           'p', 'q', 'r']
        
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id) + "\n")

    # create the dataset
    train_dataset = DynamicsDataset(data_path + "train/", 'train.h5', args.batch_size, normalize=args.normalize, 
                                    std_percentage=args.std_percentage, attitude=args.attitude, augmentations=args.augmentation)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    valid_dataset = DynamicsDataset(data_path + "valid/", 'valid.h5', args.batch_size, normalize=args.normalize, std_percentage=args.std_percentage, 
                                    attitude=args.attitude, augmentations=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # print number of datapoints
    print("Number of training datapoints:", train_dataset.X.shape[2])

    # Print shape of input and output
    print("Input shape:", train_dataset.X.shape)
    print("Output shape:", train_dataset.Y.shape)

    if args.plot == True:
        plot_data(train_dataset.X, features = INPUT_FEATURES, 
                                   save_path = experiment_path + "plots")
    print('Loading model ...')

    # Initialize the model
    model = DynamicsLearning(args, resources_path, experiment_path,
                             input_size=train_dataset.X.shape[0],
                             output_size=len(OUTPUT_FEATURES),
                             num_layers=args.mlp_layers,
                             train_steps=train_dataset.num_steps,
                             valid_steps=valid_dataset.num_steps)
    trainer = pytorch_lightning.Trainer(accelerator="gpu", devices=args.num_devices, 
                                        max_epochs=args.epochs,val_check_interval=args.val_freq, 
                                        default_root_dir=experiment_path)
    trainer.fit(model, train_dataloader, valid_dataloader)  


    
    
