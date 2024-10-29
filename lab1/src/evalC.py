import argparse

import yaml
import os
import dataclasses
import numpy as np

from pathlib import Path

from utils import (
    TrainConfigC,
    load,
    get_date_str,
)

from submission import (
    data_preprocessing_classification,
    data_split_classification,
    LogisticRegression,
    eval_LogisticRegression,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, default=None)

    args = parser.parse_args()

    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = TrainConfigC(**yaml.safe_load(f))

    np.random.seed(cfg.seed)

    if cfg.task == "Regression":
        raise ValueError("Use evalR.py for Regression")
    elif cfg.task == "Classification":
        model = LogisticRegression(cfg.in_features)
        model.load_from_state_dict(load(Path(args.results_path) / "model.pkl"))
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    # Load the dataset
    dataset = data_preprocessing_classification(cfg.data_dir, cfg.mean)

    train_set, test_set = data_split_classification(dataset)

    test_set = test_set.to_pandas().drop(columns=["__index_level_0__"])

    results_path = Path(cfg.results_path + f"_{cfg.task}") / get_date_str()
    os.makedirs(results_path, exist_ok=True)

    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(dataclasses.asdict(cfg), f)

    accuracy = eval_LogisticRegression(model, test_set.values)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
