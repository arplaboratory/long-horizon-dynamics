from torch.utils.data import DataLoader
from utils import check_folder_paths, plot_data
from config import parse_args, save_args
from data import DynamicsDataset
import pytorch_lightning
from lighting import DynamicsLearning

import sys
import time
import os

INPUT_FEATURES = {
    "euler": 13,
    "quaternion": 14,
    "rotation": 19,
}


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # Set global paths
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/"
    experiment_path = resources_path + "experiments/" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(args.run_id) + "/"

    check_folder_paths([os.path.join(experiment_path, "checkpoints"), os.path.join(experiment_path, "plots")])

    # save arguments
    save_args(args, os.path.join(experiment_path, "args.txt"))


    # Device
    args.device = "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print("Training model on cuda:" + str(args.gpu_id) + "\n")

    # Create datasets and dataloaders
    datasets = {}
    dataloaders = {}
    dataset_names = ["train", "valid"]
    for dataset_name in dataset_names:
        dataset = DynamicsDataset(os.path.join(data_path, dataset_name), f"{dataset_name}.h5", args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        datasets[dataset_name] = dataset
        dataloaders[dataset_name] = dataloader
        print(f"{dataset_name.capitalize()} shape: {dataset.X_shape}")

    # Load model
    print('Loading model ...')
    sample_data = next(iter(dataloaders["valid"]))

    # Initialize the model
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=INPUT_FEATURES[args.attitude],
        output_size=INPUT_FEATURES[args.attitude]-4,
        num_layers=args.mlp_layers,
        sample_data=sample_data,
        train_steps=datasets["train"].num_steps,
        valid_steps=datasets["valid"].num_steps
    )

    # Train the model
    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=args.epochs,
        val_check_interval=args.val_freq,
        default_root_dir=experiment_path
    )

    trainer.fit(model, dataloaders["train"], dataloaders["valid"])