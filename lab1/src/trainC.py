import argparse

import yaml
import os
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from tqdm import tqdm


from utils import (
    TrainConfigC,
    Loss,
    GD,
    DataLoader,
    Parameter,
    save,
    load,
    init_config_from_args,
)

from submission import (
    data_preprocessing_classification,
    data_split_classification,
    LogisticRegression,
    BCELoss,
    TrainerC,
)


def main():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument("--task", type=str, default="Classification")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Rosykunai/SGEMM_GPU_performance",
        help="The path to the training data.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help="The mean value to classify the data.",
    )

    # Model
    parser.add_argument(
        "--in_features",
        type=int,
        default=14,
        help="The number of input features.",
    )

    # Optimization
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
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
        "--steps",
        type=int,
        default=100,
        help="The number of optimization steps to perform.",
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
    cfg = init_config_from_args(TrainConfigC, args)

    # Initialize the model
    if cfg.task == "Regression":
        raise ValueError("Use trainR.py for Regression")
    elif cfg.task == "Classification":
        model = LogisticRegression(cfg.in_features)
    else:
        raise ValueError(f"Task {cfg.task} not supported")

    # Load the dataset
    dataset = data_preprocessing_classification(cfg.data_dir, cfg.mean)

    train_set, val_set = data_split_classification(dataset)

    results_path = Path(cfg.results_path + f"_{cfg.task}") if cfg.results_path else Path("results")

    train_set = train_set.to_pandas().drop(columns=["__index_level_0__"])

    # Train
    print("***** Running training *****")
    print(f"  Task = {cfg.task}")
    print(f"  Num examples = {len(train_set)}")
    print(f"  Total optimization steps = {cfg.steps}")

    TrainerC(
        model=model,
        dataset=train_set.values,
        loss=BCELoss(),
        optimizer=GD(model.parameters(), lr=cfg.lr, lr_decay=cfg.lr_decay, decay_every=cfg.decay_every),
        config=cfg,
        results_path=results_path,
    ).train()


if __name__ == "__main__":
    main()
