import argparse
import yaml
from model import AE
from datasets import load_from_disk
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import TrainConfig, ae_encode

from submission import PCA, GMM


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--cluster_label", action="store_true")

    args = parser.parse_args()

    dataset = load_from_disk("../mnist_encoded")
    trainset = dataset["train"]

    traindata_raw2d = np.stack(trainset["image2D"])
    traindata_raw1d = np.stack(trainset["image1D"])

    ae = AE.from_pretrained("Rosykunai/mnist-ae")
    traindata_ae = ae_encode(ae, traindata_raw2d)

    print("tSNE fitting ...")
    tsne = TSNE(2)
    traindata_tsne = tsne.fit_transform(traindata_raw1d)

    print("PCA fitting ...")
    pca = PCA_sklearn(2)
    traindata_pcask = pca.fit_transform(traindata_raw1d)

    if not args.cluster_label:
        tag = "true"
        labels = trainset["label"]
    else:
        tag = "cluster"
        with open(Path(args.results_path) / "config.yaml", "r") as f:
            cfg = TrainConfig(**yaml.safe_load(f))
        if cfg.use_pca:
            pca = PCA.from_pretrained(args.results_path + "/pca")
            traindata_pca = pca.transform(traindata_raw1d)
            gmm = GMM.from_pretrained(args.results_path + "/gmm")
            labels = gmm.predict(traindata_pca)
        else:
            gmm = GMM.from_pretrained(args.results_path + "/gmm")
            labels = gmm.predict(traindata_ae)

    plt.scatter(traindata_ae[:, 0], traindata_ae[:, 1], c=labels, cmap="tab10")
    plt.colorbar()
    plt.savefig(Path(args.results_path) / (tag + "_ae.png"))
    plt.close()

    plt.scatter(traindata_pcask[:, 0], traindata_pcask[:, 1], c=labels, cmap="tab10")
    plt.colorbar()
    plt.savefig(Path(args.results_path) / (tag + "_pca.png"))
    plt.close()

    plt.scatter(traindata_tsne[:, 0], traindata_tsne[:, 1], c=labels, cmap="tab10")
    plt.colorbar()
    plt.savefig(Path(args.results_path) / (tag + "_tsne.png"))
    plt.close()


if __name__ == "__main__":
    main()
