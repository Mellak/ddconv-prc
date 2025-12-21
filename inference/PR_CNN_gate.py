#!/usr/bin/env python3
"""
only_fastpr_operator.py
Hybrid Positron Range FORWARD operator with persistent caching.

- FORWARD model: spreads each active voxel to neighbors (conv_transpose style).
- Assumes the input μ-map is ALREADY SEGMENTED (values ≈ {Lung, Water, RibBone}) -- But works for continuous mu-values.
  * Any voxel strictly below Lung is treated as Lung (clamped).
  * All voxels are snapped to the nearest canonical value to avoid float drift.

- Caches (per volume, per kernel size K):
    * material_map.raw         (μ snapped to {Lung, Water, RibBone})
    * hom_map_K{K}.pt          (bool [3,D,H,W] masks for homogeneous regions)
    * hetero_idx_K{K}.pt       (Long [N,3] indices where heterogeneous)
- Keeps per-material kernel cache:
    * kernels_generated/kernel_{mu:.4f}_{K}.raw

CLI
---
python only_fastpr_operator.py \
  --mu /path/MuMap905.bin \
  --act /path/Prods905.raw \
  --dim 100,200,200 \
  --out /path/out_forward.bin \
  --weights /path/fpred_model_31.pth \
  --kernel-size 31 \
  --kernel-dir /path/kernels_generated \
  --cache-dir /path/cache_dir \
  --batch 400 \
  --device cuda

Notes
-----
- The μ segmentation comes from clustering/assignment to MATERIALS.
- Activity is never cached; only μ-derived artifacts are.

Author: Youness MELLAK (2025) – refactored with CLI + cache
"""
import os, sys, time, json, hashlib, argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

# --- local modules ---
from model import FilterPredictor
from utils import make_distance_grid, save_raw_3d

# ----------------------------- Defaults ---------------------------------
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

# Canonical materials (must be exactly these 3)
MATERIALS = {"Lung": 0.0247, "Water": 0.0959, "RibBone": 0.171}
MAT_ORDER = ["Lung", "Water", "RibBone"]  # fixed order for mask channels
CANONICAL = torch.tensor([MATERIALS[m] for m in MAT_ORDER], dtype=torch.float32)
# ------------------------------------------------------------------------


def parse_dim(s: str) -> Tuple[int, int, int]:
    z, y, x = [int(t) for t in s.split(",")]
    return (z, y, x)


def sha1_of_file(path: str, max_bytes: int = 4 * 1024 * 1024) -> str:
    """Small hash of the beginning of a file to invalidate cache if μ changes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        chunk = f.read(max_bytes)
    h.update(chunk)
    return h.hexdigest()


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_mu(mu_path: str, dim: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """
    Load segmented μ-map (float32 raw) and move to device.
    No thresholding here; we will snap to the canonical set afterward.
    """
    mu_np = np.fromfile(mu_path, np.float32).reshape(dim)
    mu = torch.from_numpy(mu_np).to(device)
    return mu


def snap_mu_to_materials(mu: torch.Tensor) -> torch.Tensor:
    """
    Snap μ to nearest of {Lung, Water, RibBone}, with 'below lung → lung' rule.

    Steps:
      1) Clamp any μ < Lung to Lung
      2) Compute L2 distance to each canonical value and pick the nearest
    """
    lung_val = MATERIALS["Lung"]
    vals = mu.clone()
    vals = torch.maximum(vals, torch.tensor(lung_val, device=mu.device, dtype=mu.dtype))
    # nearest neighbor among canonical values
    canon = CANONICAL.to(mu.device, mu.dtype)  # [3]
    # broadcast distances: [D,H,W,3]
    d = torch.abs(vals.unsqueeze(-1) - canon)
    idx = torch.argmin(d, dim=-1)  # [D,H,W] in {0,1,2}
    snapped = canon[idx]
    return snapped


def cache_paths(cache_dir: str, mu_tag: str, dim: Tuple[int, int, int], K: int) -> Dict[str, str]:
    dz, dy, dx = dim
    base = os.path.join(cache_dir, f"mu-{mu_tag}_D{dz}x{dy}x{dx}")
    return {
        "mat": f"{base}_material_map.raw",
        "meta": f"{base}_meta.json",
        "hom": f"{base}_hom_map_K{K}.pt",         # bool [3,D,H,W] order = MAT_ORDER
        "het_idx": f"{base}_hetero_idx_K{K}.pt",  # long [N,3]
    }


def save_material_map(path: str, mat: torch.Tensor):
    mat.detach().cpu().numpy().astype(np.float32).tofile(path)


def load_material_map(path: str, dim: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, np.float32).reshape(dim)).to(device)


def compute_homogeneity_map(material_image: torch.Tensor,
                            receptive_field: Tuple[int, int, int],
                            device: torch.device) -> torch.Tensor:
    """
    Return per-material homogeneity masks (bool [3,D,H,W]) for MAT_ORDER.

    Homogeneous at voxel p for material m if local window is (≈) entirely that material.
    We test with an average filter and threshold 0.99.
    """
    padding = tuple(k // 2 for k in receptive_field)
    D, H, W = material_image.shape
    homogeneity_maps = torch.zeros((len(MAT_ORDER), D, H, W), device=device, dtype=torch.bool)
    k = torch.ones(1, 1, *receptive_field, device=device) / float(np.prod(receptive_field))

    for idx, mat_name in enumerate(MAT_ORDER):
        mu_val = MATERIALS[mat_name]
        mask = torch.isclose(material_image, torch.tensor(mu_val, device=device, dtype=material_image.dtype), atol=1e-5).float()
        # Using conv3d on mask gives the fraction of that material in the window
        conv = F.conv3d(mask.unsqueeze(0).unsqueeze(0), k, padding=padding).squeeze(0).squeeze(0)
        homogeneity_maps[idx] = conv >= 0.99
    return homogeneity_maps


def load_or_generate_kernel(mu_value: float, model: torch.nn.Module, kernel_dir: str,
                            kernel_size: int, device: torch.device) -> torch.Tensor:
    """Per-material kernel cache."""
    ensure_dirs(kernel_dir)
    fname = os.path.join(kernel_dir, f"kernel_{mu_value:.4f}_{kernel_size}.raw")
    if os.path.exists(fname):
        k_np = np.fromfile(fname, np.float32).reshape(kernel_size, kernel_size, kernel_size)
        return torch.from_numpy(k_np).float().unsqueeze(0).unsqueeze(0).to(device)
    # generate
    mu_patch = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=device) * mu_value
    dist = make_distance_grid(kernel_size, device=device, normalize="max", negate=True)
    with torch.no_grad(), autocast():
        pred = model(mu_patch, dist).view(1, -1)
        pred = F.softmax(pred, dim=1).view(1, 1, kernel_size, kernel_size, kernel_size)
    k = pred / (pred.sum() + 1e-12)
    k.detach().cpu().numpy().astype(np.float32).tofile(fname)
    return k


def apply_kernels_convolution_forward(prod_image: torch.Tensor,
                                      kernels: Dict[str, torch.Tensor],
                                      homogeneous_mask: torch.Tensor,
                                      receptive_field: Tuple[int, int, int],
                                      device: torch.device) -> torch.Tensor:
    """
    Forward op on homogeneous regions: conv_transpose3d with the material kernel,
    gated by each material’s homogeneous mask.
    """
    padding = tuple(k // 2 for k in receptive_field)
    out = torch.zeros_like(prod_image, device=device)
    for idx, mat_name in enumerate(MAT_ORDER):
        kernel = kernels[mat_name]  # [1,1,K,K,K]
        filt = homogeneous_mask[idx] * prod_image  # [D,H,W]
        conv = F.conv_transpose3d(filt.unsqueeze(0).unsqueeze(0), kernel, padding=padding)  # [1,1,D,H,W]
        out += conv.squeeze(0).squeeze(0)
    return out


@torch.no_grad()
def simulate_heterogeneous_forward(mu_image: torch.Tensor,
                                   prod_image: torch.Tensor,
                                   hetero_idx: torch.Tensor,
                                   model: torch.nn.Module,
                                   kernel_size: int,
                                   batch_size: int,
                                   device: torch.device) -> torch.Tensor:
    """
    Forward op on heterogeneous voxels:
    - predict kernel per active voxel
    - scatter_add contributions over neighbors (spread).
    """
    D, H, W = prod_image.shape
    pad = kernel_size // 2
    offsets = torch.arange(-pad, pad + 1, device=device)
    dz, dy, dx = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    dz, dy, dx = dz.flatten(), dy.flatten(), dx.flatten()
    dist = make_distance_grid(kernel_size, device=device, normalize="max", negate=True)
    out = torch.zeros_like(prod_image, device=device).view(-1)

    if hetero_idx.numel() == 0:
        return out.view(D, H, W)

    model.eval()
    for start in tqdm(range(0, hetero_idx.size(0), batch_size), desc="[hetero FWD] batches"):
        end = min(start + batch_size, hetero_idx.size(0))
        b_idx = hetero_idx[start:end]  # [B,3]
        B = b_idx.size(0)
        z, y, x = b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]
        z_b, y_b, x_b = (z[:, None] + dz).clamp(0, D - 1), (y[:, None] + dy).clamp(0, H - 1), (x[:, None] + dx).clamp(0, W - 1)
        mu_win = mu_image[z_b, y_b, x_b].view(B, 1, kernel_size, kernel_size, kernel_size)
        act_val = prod_image[z, y, x].view(B, 1)
        with autocast():
            k_pred = model(mu_win, dist).view(B, -1)
            k_pred = F.softmax(k_pred, dim=1)
        lin_idx = (z_b * H * W + y_b * W + x_b).view(-1)
        spread = (k_pred * act_val).view(-1)
        out.scatter_add_(0, lin_idx, spread)
    return out.view(D, H, W)


def simulate_pr_forward(mu_path: str,
                        act_path: str,
                        dim: Tuple[int, int, int],
                        out_path: str,
                        weights: str,
                        kernel_size: int,
                        kernel_dir: str,
                        cache_dir: str,
                        batch: int,
                        device_str: str):
    device = torch.device(device_str)
    ensure_dirs(os.path.dirname(out_path), kernel_dir, cache_dir)

    mu_tag = sha1_of_file(mu_path)
    cpaths = cache_paths(cache_dir, mu_tag, dim, kernel_size)

    # ----- Load μ
    mu = load_mu(mu_path, dim, device)

    # ----- Material map cache (snap to canonical)
    need_material = True
    if os.path.exists(cpaths["mat"]) and os.path.exists(cpaths["meta"]):
        try:
            meta = json.load(open(cpaths["meta"], "r"))
            if meta.get("mu_tag") == mu_tag and meta.get("dim") == list(dim):
                mat = load_material_map(cpaths["mat"], dim, device)
                need_material = False
        except Exception:
            pass

    if need_material:
        mat = snap_mu_to_materials(mu)  # snap & clamp
        save_material_map(cpaths["mat"], mat)
        json.dump({"mu_tag": mu_tag, "dim": list(dim)}, open(cpaths["meta"], "w"))

    # ----- Homogeneity / heterogeneous cache (per K)
    if os.path.exists(cpaths["hom"]) and os.path.exists(cpaths["het_idx"]):
        hom_map = torch.load(cpaths["hom"], map_location=device)
        hetero_idx = torch.load(cpaths["het_idx"], map_location=device)
    else:
        hom_map = compute_homogeneity_map(mat, (kernel_size,) * 3, device)
        hetero_mask = (~torch.any(hom_map, dim=0))  # [D,H,W]
        hetero_idx = torch.nonzero(hetero_mask, as_tuple=False)  # [N,3]
        torch.save(hom_map, cpaths["hom"])
        torch.save(hetero_idx, cpaths["het_idx"])

    # ----- Model + per-material kernels
    model = FilterPredictor(input_channel=2, in_between_channel=16, output_channel=1, num_residual_blocks=4).to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    kernels = {name: load_or_generate_kernel(MATERIALS[name], model, kernel_dir, kernel_size, device)
               for name in MAT_ORDER}

    # ----- Load activity (changes every iteration)
    prod_np = np.fromfile(act_path, np.float32).reshape(dim)
    prod = torch.from_numpy(prod_np).to(device)

    # ----- Homogeneous forward (convT)
    out_hom = apply_kernels_convolution_forward(prod, kernels, hom_map, (kernel_size,) * 3, device)

    '''# ----- Heterogeneous forward
    # Restrict to positive activity for efficiency
    pos_mask = prod > 0
    het_pos_idx = hetero_idx[pos_mask[hetero_idx[:, 0], hetero_idx[:, 1], hetero_idx[:, 2]]]
    # use μ snapped (mat) rather than raw μ for robustness'''
    out_hetero = simulate_heterogeneous_forward(mat, prod, hetero_idx, model, kernel_size, batch, device)

    out = out_hom + out_hetero

    # ----- Save
    #save_raw_3d(out_path, out.detach().cpu().numpy())
    return out


def build_argparser():
    p = argparse.ArgumentParser(description="Hybrid PR FORWARD operator with caching (conv_transpose+model).")
    p.add_argument("--mu", required=True, help="Path to segmented μ-map (float32 raw).")
    p.add_argument("--act", required=True, help="Path to activity / Prods (float32 raw).")
    p.add_argument("--dim", required=True, help="Grid dims 'Z,Y,X'.")
    p.add_argument("--out", required=True, help="Output path (float32 raw).")
    p.add_argument("--weights", required=True, help="Model weights .pth")
    p.add_argument("--kernel-size", type=int, default=31)
    p.add_argument("--kernel-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--batch", type=int, default=400)
    p.add_argument("--device", default=DEVICE_DEFAULT)
    return p



if __name__ == "__main__":
    import sys, time

    # ============ CASToR-compatible entry ============
    if len(sys.argv) < 2:
        print("Usage (from CASToR or manually): python only_fastpr_Toperator.py <activity_image>")
        sys.exit(1)

    # ----- fixed configuration -----
    path_act_image = sys.argv[2]
    path_mat_image = "/home/youness/data/PR_Correction/Reconstruction/SimPhantom/data/MuMap905.bin"

    # image grid (Z,Y,X)
    dimZ, dimY, dimX = 100, 200, 200
    DIM = (dimZ, dimY, dimX)
    expected_size = dimX * dimY * dimZ

    kernel_size = 11
    weights_path = f"/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/Weights/fpred_model_{kernel_size}.pth"
    kernel_dir  = "/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/tmp/kernels_generated"
    cache_dir   = "/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/tmp/cache"
    out_path    = "/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/tmp/result_image.bin"

    # sanity check
    if not os.path.exists(path_act_image):
        print(f"[error] Activity image not found: {path_act_image}")
        sys.exit(1)

    t0 = time.time()
    out = simulate_pr_forward(
        mu_path     = path_mat_image,
        act_path    = path_act_image,
        dim         = DIM,
        out_path    = out_path,
        weights     = weights_path,
        kernel_size = kernel_size,
        kernel_dir  = kernel_dir,
        cache_dir   = cache_dir,
        batch       = 400,
        device_str  = DEVICE_DEFAULT,
    )
    print(f"[done] time={time.time()-t0:.1f}s")
    

    out = out.cpu().numpy()
    out.tofile("/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/result_image.bin")
    print("Saved result_image to result_image.bin")

    '''# save the material_image as a .bin file
    material_image = material_image.cpu().numpy()
    material_image.tofile("/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/material_image.bin")
    print("Saved material_image to material_image.bin")

    # save the activity_image as a .bin file
    activity_image = activity_image.cpu().numpy()
    activity_image.tofile("/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/activity_image.bin")
    print("Saved activity_image to activity_image.bin")'''


