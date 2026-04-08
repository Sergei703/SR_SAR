import os
import numpy as np
import torch
from torch import nn
from PIL import Image


class RRDB(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )

    def forward(self, x):
        return x + 0.2 * self.block(x)


class Generator(nn.Module):
    def __init__(self, num_rrdb=16, nf=64):
        super().__init__()
        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2),
        )
        self.conv_last = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        body = self.conv_body(self.body(fea))
        fea = fea + body
        fea = self.upsample(fea)
        out = self.conv_last(fea)
        return torch.tanh(out)


def load_model(path: str, device: torch.device) -> Generator:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = Generator().to(device)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    arr = tensor.squeeze(0).squeeze(0).numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def tensor_to_numpy01(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    return tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)


def numpy01_to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
