# utils_processing.py
import torch
import numpy as np
from typing import Literal, Tuple

def make_distance_grid(size: int,
                       device: torch.device,
                       dtype=torch.float32,
                       negate: bool = True,
                       normalize: Literal["max","none"] = "max"):
    c = size // 2
    z, y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype),
        indexing="ij"
    )
    dist = torch.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)
    if normalize == "max":
        dmax = dist.max().clamp_min(1e-8)
        dist = dist / dmax
    if negate:
        dist = -dist
    return dist.unsqueeze(0).unsqueeze(0)  # [1,1,N,N,N]

def normalize_array(arr: np.ndarray, mode: str = "sum", eps: float = 1e-12):
    if mode == "sum":
        s = float(arr.sum())
        return (arr / s) if s > eps else np.zeros_like(arr)
    elif mode == "max":
        m = float(np.max(np.abs(arr)))
        return (arr / m) if m > eps else np.zeros_like(arr)
    else:
        raise ValueError(f"Unsupported norm_mode: {mode}")

def nearest_rep_value(mu_val: float, reps: Tuple[float,float,float]):
    return min(reps, key=lambda r: abs(mu_val - r))



def load_raw_3d(path: str, shape):
    D,H,W = shape
    n = D*H*W
    with open(path, "rb") as f:
        arr = np.fromfile(f, dtype=np.float32, count=n)
    if arr.size != n:
        raise ValueError(f"{path}: got {arr.size} floats, expected {n} for {shape}")
    return arr.reshape(shape).astype(np.float32)

def save_raw_3d(path: str, arr: np.ndarray):
    np.asarray(arr, dtype=np.float32).tofile(path)

def pad_to_expected(flat: np.ndarray, expected: int):
    if flat.size < expected:
        pad = expected - flat.size
        flat = np.pad(flat, (0,pad), mode="constant")
    return flat

def to_torch_3d(arr: np.ndarray, device=None, dtype=torch.float32):
    t = torch.as_tensor(arr, dtype=dtype)
    return t.to(device) if device is not None else t
