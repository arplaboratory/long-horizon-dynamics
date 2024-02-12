import torch 
from torch.utils.data import DataLoader
import warnings
from dynamics_learning.utils import check_folder_paths
from config import parse_args, save_args
from dynamics_learning.data import load_dataset
import pytorch_lightning
from dynamics_learning.lighting import DynamicsLearning
from config import load_args

import sys
import time
import os
import glob

warnings.filterwarnings('ignore')


#----------------------------------------------------------------------------

def main(args, hdf5_files, model_path, plotting_data_path, model_name):

    # WandB Logging
    wandb_logger = pytorch_lightning.loggers.WandbLogger(name="wandb_logger", project="dynamics_learning", save_dir=experiment_path) 

    
    # Load datasets from hdf5 files and return a dictionary of dataloaders
    test_dataloaders = [load_dataset("test", data_path, hdf5_file, args, 0, False)[1] for hdf5_file in hdf5_files]

    dataloaders = {}
    for hdf5_file in hdf5_files:
        dataset, dataloader = load_dataset("test", data_path, hdf5_file, args, 0, False)
        # GEt name of the file
        file_name = hdf5_file.split("/")[-1].split(".")[0]

        dataloaders[file_name] = dataloader

    # Load model
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=14,
        output_size=10,
        valid_data=None,
        max_iterations=1,
    )


    # # Load the model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    trainer = pytorch_lightning.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=0,
    )

    # Test model on different trajectories and display trajectory names in the logs
    trainer.test(model, test_dataloaders, verbose=True)

    print(dataloaders.keys())



if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # Asser model type
    assert args.model_type in ["mlp", "lstm", "gru", "tcn", "transformer"], "Model type must be one of [mlp, lstm, gru, tcn, transformer]"

    # Assert attitude type
    assert args.attitude in ["euler", "quaternion", "rotation"], "Attitude type must be one of [euler, quaternion, rotation]"

    # Seed
    pytorch_lightning.seed_everything(args.seed)

    # Assert vehicle type
    assert args.vehicle_type in ["fixed_wing", "quadrotor", "neurobem"], "Vehicle type must be one of [fixed_wing, quadrotor, neurobem]"
    if args.vehicle_type == "fixed_wing":
        vehicle_type = "fixed_wing"
    elif args.vehicle_type == "quadrotor":
        vehicle_type = "quadrotor"
    elif args.vehicle_type == "neurobem":
        vehicle_type = "neurobem"

    # Set global paths
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/" + vehicle_type + "/"
    experiment_path = experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
    model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)
    plotting_data_path = experiment_path + "plotting_data/"

    check_folder_paths([os.path.join(experiment_path, "checkpoints"), os.path.join(experiment_path, "plots"), os.path.join(experiment_path, "plots", "trajectory"), 
                        os.path.join(experiment_path, "plots", "testset")])

    print(experiment_path)
    print("Testing Dynamics model:", model_path)
    args = load_args(experiment_path + "args.txt")

    model_name = args.model_type + "_" + str(args.history_length) + "_" + args.vehicle_type + "_" + str(args.unroll_length) 

    args.unroll_length = 60
    
    # device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


    # Load all hdf5 paths in the data folder into a list sorted 
    hdf5_files = sorted(glob.glob(data_path + "test/*.h5"))

    # Test model on different trajectories
    main(args, hdf5_files, model_path, plotting_data_path, model_name)