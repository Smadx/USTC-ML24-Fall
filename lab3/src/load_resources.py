from datasets import DatasetDict, load_dataset
from model import AE, ClassUNet
import numpy as np
import torch
import os

# If downloading the dataset is slow, you can use the mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Load the dataset
raw_dataset = load_dataset("ylecun/mnist")

trainset = raw_dataset["train"]
testset = raw_dataset["test"]


# Encode the image
def encode_image(example):
    img_np = np.array(example["image"], dtype=np.uint8)

    img_np_with_channel = np.expand_dims(img_np, axis=0)
    img_np_flat = img_np.flatten()
    example["image2D"] = np.array(img_np_with_channel, dtype=np.uint8)
    example["image1D"] = np.array(img_np_flat, dtype=np.uint8)
    return example


trainset = trainset.map(encode_image)
testset = testset.map(encode_image)

trainset.set_format(type="numpy", columns=["image2D", "image1D"])
testset.set_format(type="numpy", columns=["image2D", "image1D"])

dataset = DatasetDict({"train": trainset, "test": testset})
dataset.save_to_disk("../mnist_encoded")

print("Dataset saved to disk")

# Load model
mnist_ae = AE.from_pretrained("Rosykunai/mnist-ae")
mnist_ddpm = ClassUNet.from_pretrained("Rosykunai/mnist-ddpm")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Model loaded, {device} used")
