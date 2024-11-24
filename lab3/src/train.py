import argparse
import dataclasses
import yaml
from datasets import load_from_disk
from model import AE
import numpy as np

from utils import (
    TrainConfig,
    ae_encode,
    init_config_from_args,
    handle_results_path,
)

from submission import GMM, PCA


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dim", type=int, default=10)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=100)

    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = init_config_from_args(TrainConfig, args)

    results_path = handle_results_path(cfg.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(dataclasses.asdict(cfg), f)

    if cfg.use_pca:
        dataset = load_from_disk("../mnist_encoded")
        trainset = dataset["train"].to_pandas()
        traindata_raw = np.vstack(trainset["image1D"].to_numpy())
        pca = PCA(dim=cfg.embedding_dim)
        pca.fit(traindata_raw)
        traindata = pca.transform(traindata_raw)
        pca.save_pretrained(results_path / "pca")
        print(f"Succesfully saved PCA model to {results_path}/pca")
    else:
        dataset = load_from_disk("../mnist_encoded")
        trainset = dataset["train"]
        traindata_raw = np.stack(trainset["image2D"])
        ae = AE.from_pretrained("Rosykunai/mnist-ae")
        traindata = ae_encode(ae, traindata_raw)

    gmm_dim = cfg.embedding_dim if cfg.use_pca else 2

    gmm = GMM(n_components=cfg.n_components, data_dim=gmm_dim)
    gmm.fit(traindata, max_iter=cfg.max_iter)
    gmm.save_pretrained(results_path / "gmm")
    print(f"Succesfully saved GMM model to {results_path}/gmm")


if __name__ == "__main__":
    main()
