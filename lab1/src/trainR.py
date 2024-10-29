import argparse

import numpy as np
from pathlib import Path

from utils import (
    TrainConfigR,
    SGD,
    init_config_from_args,
)

from submission import (
    data_preprocessing_regression,
    data_split_regression,
    LinearRegression,
    TrainerR,
    MSELoss,
)


def main():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument("--task", type=str, default="Regression")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Rosykunai/SGEMM_GPU_performance",
        help="The path to the training data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="The batch size for training.",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Whether to shuffle the data.",
    )

    # Model
    parser.add_argument(
        "--in_features",
        type=int,
        default=14,
        help="The number of input features.",
    )
    parser.add_argument(
        "--out_features",
        type=int,
        default=1,
        help="The number of output features.",
    )

    # Optimization
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-9,
        help="The learning rate used for optimization.",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.99,
        help="The learning rate decay factor when using SGD.",
    )
    parser.add_argument(
        "--decay_every",
        type=int,
        default=10,
        help="The number of steps after which to decay the learning rate.",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to train.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="The path to save the results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="The seed to use for reproducibility.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = init_config_from_args(TrainConfigR, args)

    # Initialize the model
    if cfg.task == "Regression":
        model = LinearRegression(cfg.in_features, cfg.out_features)
    elif cfg.task == "Classification":
        raise ValueError("Use trainC.py for classification")
    else:
        raise ValueError(f"Task {cfg.task} not supported")

    # Load the dataset
    dataset = data_preprocessing_regression(cfg.data_dir)

    trainloader, testloader = data_split_regression(dataset, cfg.batch_size, cfg.shuffle)

    results_path = Path(cfg.results_path + f"_{cfg.task}") if cfg.results_path else Path("results")

    # Train
    print("***** Running training *****")
    print(f"  Task = {cfg.task}")
    print(f"  Num examples = {len(trainloader.dataset)}")
    print(f"  Num batches each epoch = {len(trainloader)}")
    print(f"  Num Epochs = {cfg.epochs}")
    print(f"  Batch size = {cfg.batch_size}")
    print(f"  Total optimization steps = {len(trainloader) * cfg.epochs}")

    TrainerR(
        model=model,
        train_loader=trainloader,
        loss=MSELoss(),
        optimizer=SGD(model.parameters(), lr=cfg.lr, lr_decay=cfg.lr_decay, decay_every=cfg.decay_every),
        config=cfg,
        results_path=results_path,
    ).train()


if __name__ == "__main__":
    main()
