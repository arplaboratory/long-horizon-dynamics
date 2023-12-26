from torch.utils.data import DataLoader
import warnings
from dynamics_learning.utils import check_folder_paths
from config import parse_args, save_args
from dynamics_learning.data import load_dataset
import pytorch_lightning
from dynamics_learning.lighting import DynamicsLearning

import sys
import time
import os

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

    # Checkopoint 
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(experiment_path, "checkpoints"),
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    checkpoint_callback.CHECKPOINT_NAME_LAST = "last_model"
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Create datasets and dataloaders
    train_dataset, train_loader = load_dataset(
        "training",
        data_path + "train/",
        "train.h5",
        args,
        num_workers=16,
        pin_memory=True,
    )


    valid_dataset, valid_loader = load_dataset(
        "validation",
        data_path + "valid/",
        "valid.h5",
        args,
        num_workers=16,
        pin_memory=True,
    )

    input_size = train_dataset.X_shape[2]
    output_size = train_dataset.Y_shape[1]

    val_gt = valid_dataset.Y
    
    # Load model
    print('Loading model ...')

    # Initialize the model
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=input_size,
        output_size=output_size,
        valid_data=val_gt,
        max_iterations=train_dataset.num_steps * args.epochs,
    )

    # Train the model
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_freq,
        default_root_dir=experiment_path,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0
    )
    if trainer.is_global_zero:
        wandb_logger.experiment.config.update(vars(args))
    # trainer.validate(model, dataloaders=valid_loader)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


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
    experiment_path = resources_path + "experiments/" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(args.run_id) + "/"

    check_folder_paths([os.path.join(experiment_path, "checkpoints"), os.path.join(experiment_path, "plots"), os.path.join(experiment_path, "plots", "trajectory"), 
                        os.path.join(experiment_path, "plots", "testset")])

    # save arguments
    save_args(args, os.path.join(experiment_path, "args.txt"))


    # Device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id) + "\n")

    # Train model
    main(args)

    