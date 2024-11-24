import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from model import AE, ClassUNet
from typing import Union
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from diffusers import DDPMScheduler


@dataclass
class TrainConfig:
    embedding_dim: int
    use_pca: bool
    n_components: int
    max_iter: int
    results_path: str
    seed: int


def ae_encode(ae: AE, dataset: np.ndarray) -> np.ndarray:
    """
    Use autoencoders to encode data.

    Args:
        - ae(AE): Autoencoder
        - dataset(np.ndarray): shape is (B, C, H, W)
    """
    ae.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae.to(device)
    B, C, H, W = dataset.shape
    img_tensor = torch.from_numpy(dataset)
    img_tensor = img_tensor.to(device).float()

    encoded_list = []
    print("AE encoding ...")
    with torch.no_grad():
        for i in tqdm(range(B)):
            img = img_tensor[i].unsqueeze(0)  # [1, C, H, W]
            img_encoded = ae.encoder(img)
            encoded_list.append(img_encoded.squeeze(0))  # [dim]

    encoded_imgs = torch.stack(encoded_list)  # [B, dim]
    if device == "cuda":
        return encoded_imgs.detach().cpu().numpy()
    return encoded_imgs.numpy()


def sample_from_ddpm(label: int, path: Union[str, Path]):
    """
    Sample from DDPM model.

    Args:
        - label(int): Label for the image.
        - path(Union[str, Path]): Path to save the image.
    """
    model = ClassUNet.from_pretrained("Rosykunai/mnist-ddpm")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    noise_schedule = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    x = torch.randn(1, 1, 28, 28).to(device)
    y = torch.tensor([label]).to(device)

    for i, t in tqdm(enumerate(noise_schedule.timesteps)):
        with torch.no_grad():
            pred = model(x, t, y)
        x = noise_schedule.step(pred, t, x).prev_sample
    img = x[0][0].detach().cpu().numpy()

    img = Image.fromarray((img * 255).astype("uint8"))
    path = Path(path)
    img.save(path / "ddpm_sample.png")


def init_config_from_args(cls, args):
    """Initialize a dataclass from a Namespace."""
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def handle_results_path(res_path: str, default_root: str = "../results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path) / get_date_str()
    print(f"Results will be saved to '{results_path}'")
    return results_path
