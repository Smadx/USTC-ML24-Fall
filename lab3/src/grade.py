import argparse
import yaml
from datasets import load_from_disk
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from model import AE
from utils import TrainConfig, ae_encode, sample_from_ddpm

from submission import PCA, GMM, sample_from_gmm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--results_path", type=str, default=None, required=True)

    args = parser.parse_args()

    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    dataset = load_from_disk("../mnist_encoded")
    trainset = dataset["train"].to_pandas()
    testset = dataset["test"].to_pandas()

    traindata_raw1d = np.vstack(trainset["image1D"].to_numpy())
    testdata_raw1d = np.vstack(testset["image1D"].to_numpy())

    print("---------------------- Testing your model ----------------------")
    if cfg.use_pca:
        pca = PCA.from_pretrained(args.results_path + "/pca")
        testdata = pca.transform(testdata_raw1d)
    else:
        testset = dataset["test"]
        testdata_raw2d = np.stack(testset["image2D"])
        ae = AE.from_pretrained("Rosykunai/mnist-ae")
        testdata = ae_encode(ae, testdata_raw2d)

    gmm = GMM.from_pretrained(args.results_path + "/gmm")
    cluster_labels = gmm.predict(testdata)

    dbscore = davies_bouldin_score(testdata_raw1d, cluster_labels)
    print(f"Your model got a Davies Bouldin score of {dbscore:.2f}")
    print("--------------------- Testing sklearn model ---------------------")
    if cfg.use_pca:
        pca_sklearn = PCA_sklearn(n_components=cfg.embedding_dim, random_state=cfg.seed)
        pca_sklearn.fit(traindata_raw1d)
        traindata_sklearn = pca_sklearn.transform(traindata_raw1d)
        testdata_sklearn = pca_sklearn.transform(testdata_raw1d)
    else:
        trainset = dataset["train"]
        traindata_raw2d = np.stack(trainset["image2D"])
        testdata_raw2d = np.stack(testset["image2D"])
        ae_sklearn = AE.from_pretrained("Rosykunai/mnist-ae")
        traindata_sklearn = ae_encode(ae_sklearn, traindata_raw2d)
        testdata_sklearn = ae_encode(ae_sklearn, testdata_raw2d)

    gmm_sklearn = GaussianMixture(n_components=cfg.n_components, max_iter=cfg.max_iter, random_state=cfg.seed)
    gmm_sklearn.fit(traindata_sklearn)
    cluster_labels_sklearn = gmm_sklearn.predict(testdata_sklearn)

    dbscore_sklearn = davies_bouldin_score(testdata_raw1d, cluster_labels_sklearn)
    print(f"The sklearn model got a Davies Bouldin score of {dbscore_sklearn:.2f}")

    if cfg.use_pca:
        print("----------------------- Sampling from GMM -----------------------")
        sample_from_gmm(gmm, pca, args.sample_index, args.results_path)
        print("---------------------- Sampling from DDPM ----------------------")
        sample_from_ddpm(6, args.results_path)

    cluster_score = 30 * dbscore_sklearn / dbscore
    cluster_score = max(0, min(cluster_score, 30))

    print(f"You got a score of {cluster_score:.2f}/30 in total.")


if __name__ == "__main__":
    main()
