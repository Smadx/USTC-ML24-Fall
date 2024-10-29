import argparse

import yaml
import os
import dataclasses
import numpy as np

from pathlib import Path

from utils import (
    TrainConfigR,
    load,
    get_date_str,
)

from submission import (
    LinearRegression,
    data_preprocessing_regression,
    data_split_regression,
    eval_LinearRegression,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, default=None)

    args = parser.parse_args()

    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = TrainConfigR(**yaml.safe_load(f))

    np.random.seed(cfg.seed)

    if cfg.task == "Regression":
        model = LinearRegression(cfg.in_features, cfg.out_features)
        model.load_from_state_dict(load(Path(args.results_path) / "model.pkl"))
    elif cfg.task == "Classification":
        raise ValueError("Use evalC.py for classification")
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    dataset = data_preprocessing_regression(cfg.data_dir)

    trainloader, testloader = data_split_regression(dataset, cfg.batch_size, cfg.shuffle)

    results_path = Path(cfg.results_path) / get_date_str()

    os.makedirs(results_path, exist_ok=True)

    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(dataclasses.asdict(cfg), f)

    mu, relative_error = eval_LinearRegression(model, testloader)
    print(f"Average prediction: {mu}")
    print(f"Relative error: {relative_error}")


if __name__ == "__main__":
    main()
