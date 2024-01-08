from torch.utils.data import DataLoader
import warnings
from dynamics_learning.utils import check_folder_paths
from config import parse_args, save_args, load_args
from dynamics_learning.data import load_dataset
import pytorch_lightning
from dynamics_learning.lighting import DynamicsLearning

import sys
import os
import glob

warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------
INPUT_FEATURES = {
    "euler": 13,
    "quaternion": 14,
    "rotation": 19,
}
#----------------------------------------------------------------------------

def main(args):

    # WandB Logging
    wandb_logger = pytorch_lightning.loggers.WandbLogger(name="wandb_logger", project="dynamics_learning", save_dir=experiment_path) 

    # Create datasets and dataloaders
    test_dataset, test_loader = load_dataset(
        "validation",
        data_path + "test/",
        "test.h5",
        args,
        num_workers=16,
        pin_memory=True,
    )

    input_size = test_dataset.X_shape[2]
    output_size = 4

    test_gt = test_dataset.Y
    
    # Load model
    print('Loading model ...')

    # Initialize the model
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=input_size,
        output_size=output_size,
        valid_data=test_gt,
        max_iterations=test_dataset.num_steps * args.epochs,
    )

    # Train the model
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=experiment_path,
        logger=wandb_logger,
     
    )
    if trainer.is_global_zero:
        wandb_logger.experiment.config.update(vars(args))
    
    # Validate the model
    trainer.validate(model, test_loader, ckpt_path=model_path)

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
    assert args.vehicle_type in ["fixed_wing", "quadrotor"], "Vehicle type must be one of [fixed_wing, quadrotor]"

    if args.vehicle_type == "fixed_wing":
        vehicle_type = "fixed_wing"
    else:
        vehicle_type = "quadrotor"

    # Set global paths
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/" + vehicle_type + "/"
    experiment_path = experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime) 
    model_path = max(glob.glob(experiment_path + "checkpoints/*.pth", recursive=True), key=os.path.getctime)

    check_folder_paths([os.path.join(experiment_path, "checkpoints"), os.path.join(experiment_path, "plots"), os.path.join(experiment_path, "plots", "trajectory"), 
                        os.path.join(experiment_path, "plots", "testset")])

    print(experiment_path)
    print("Testing Dynamics model:", model_path)
    args = load_args(experiment_path + "args.txt")

    # Set unroll length to 1
    args.unroll_length = 1
    

    # Device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id) + "\n")

    # Train model
    main(args)

    