#!/usr/bin/env python3
"""
only_fastpr_Toperator.py
Hybrid Positron Range TRANSPOSE operator with persistent caching.

- TRANSPOSE model: gathers neighborhood into center voxel (conv3d style).
- Assumes the input μ-map is ALREADY SEGMENTED (values ≈ {Lung, Water, RibBone}).
  * Any voxel strictly below Lung is treated as Lung (clamped).
  * All voxels are snapped to the nearest canonical value to avoid float drift.

- Caches (per volume, per kernel size K):
    * material_map.raw         (μ snapped to {Lung, Water, RibBone})
    * hom_map_K{K}.pt          (bool [3,D,H,W] masks for homogeneous regions; order = MAT_ORDER)
    * hetero_idx_K{K}.pt       (Long [N,3] indices where heterogeneous)
- Keeps per-material kernel cache:
    * kernels_generated/kernel_{mu:.4f}_{K}.raw

CLI
---
python only_fastpr_Toperator.py \
  --mu /path/MuMap905.bin \
  --act /path/SomeInput.raw \
  --dim 100,200,200 \
  --out /path/out_transpose.bin \
  --weights /path/fpred_model_31.pth \
  --kernel-size 31 \
  --kernel-dir /path/kernels_generated \
  --cache-dir /path/cache_dir \
  --batch 400 \
  --device cuda
  
Author: Youness MELLAK (2025)
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
MAT_ORDER = ["Lung", "Water", "RibBone"]  # fixed channel order
CANONICAL = torch.tensor([MATERIALS[m] for m in MAT_ORDER], dtype=torch.float32)
# ------------------------------------------------------------------------


def parse_dim(s: str) -> Tuple[int, int, int]:
    z, y, x = [int(t) for t in s.split(",")]
    return (z, y, x)


def sha1_of_file(path: str, max_bytes: int = 4 * 1024 * 1024) -> str:
    """Small hash of the beginning of a file to invalidate cache if μ changes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_mu(mu_path: str, dim: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """Load segmented μ-map (float32 raw) and move to device."""
    mu_np = np.fromfile(mu_path, np.float32).reshape(dim)
    return torch.from_numpy(mu_np).to(device)


def snap_mu_to_materials(mu: torch.Tensor) -> torch.Tensor:
    """
    Snap μ to nearest of {Lung, Water, RibBone}, with 'below lung → lung' rule.

    Steps:
      1) Clamp any μ < Lung to Lung
      2) Compute L1 distance to each canonical value and pick the nearest
    """
    lung_val = MATERIALS["Lung"]
    vals = torch.maximum(mu, torch.tensor(lung_val, device=mu.device, dtype=mu.dtype))
    canon = CANONICAL.to(mu.device, mu.dtype)  # [3]
    d = torch.abs(vals.unsqueeze(-1) - canon)  # [D,H,W,3]
    idx = torch.argmin(d, dim=-1)              # [D,H,W] in {0,1,2}
    return canon[idx]


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
    Per-material homogeneity masks (bool [3,D,H,W]) for MAT_ORDER.
    Homogeneous if the local window is ≥99% that material.
    """
    padding = tuple(k // 2 for k in receptive_field)
    D, H, W = material_image.shape
    homogeneity_maps = torch.zeros((len(MAT_ORDER), D, H, W), device=device, dtype=torch.bool)
    k = torch.ones(1, 1, *receptive_field, device=device) / float(np.prod(receptive_field))

    for idx, mat_name in enumerate(MAT_ORDER):
        mu_val = MATERIALS[mat_name]
        mask = torch.isclose(material_image, torch.tensor(mu_val, device=device, dtype=material_image.dtype), atol=1e-5).float()
        frac = F.conv3d(mask.unsqueeze(0).unsqueeze(0), k, padding=padding).squeeze(0).squeeze(0)
        homogeneity_maps[idx] = frac >= 0.99
    return homogeneity_maps


def load_or_generate_kernel(mu_value: float, model: torch.nn.Module, kernel_dir: str,
                            kernel_size: int, device: torch.device) -> torch.Tensor:
    """Per-material kernel cache."""
    ensure_dirs(kernel_dir)
    fname = os.path.join(kernel_dir, f"kernel_{mu_value:.4f}_{kernel_size}.raw")
    if os.path.exists(fname):
        k_np = np.fromfile(fname, np.float32).reshape(kernel_size, kernel_size, kernel_size)
        return torch.from_numpy(k_np).float().unsqueeze(0).unsqueeze(0).to(device)

    # Generate from model once per material
    mu_patch = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=device) * mu_value
    dist = make_distance_grid(kernel_size, device=device, normalize="max", negate=True)
    with torch.no_grad(), autocast():
        pred = model(mu_patch, dist).view(1, -1)
        pred = F.softmax(pred, dim=1).view(1, 1, kernel_size, kernel_size, kernel_size)
    k = pred / (pred.sum() + 1e-12)
    k.detach().cpu().numpy().astype(np.float32).tofile(fname)
    return k


def apply_kernels_convolution_transpose(prod_image: torch.Tensor,
                                        kernels: Dict[str, torch.Tensor],
                                        homogeneous_mask: torch.Tensor,
                                        receptive_field: Tuple[int, int, int],
                                        device: torch.device) -> torch.Tensor:
    """
    Homogeneous transpose: conv3d (gather) with the per-material kernel,
    multiplied by the material’s homogeneity mask (so we only trust it there).
    """
    padding = tuple(k // 2 for k in receptive_field)
    out = torch.zeros_like(prod_image, device=device)
    for idx, mat_name in enumerate(MAT_ORDER):
        kernel = kernels[mat_name]  # [1,1,K,K,K]
        conv = F.conv3d(prod_image.unsqueeze(0).unsqueeze(0), kernel, padding=padding).squeeze(0).squeeze(0)
        out += conv * homogeneous_mask[idx]
    return out


@torch.no_grad()
def simulate_heterogeneous_transpose(mu_image: torch.Tensor,
                                     prod_image: torch.Tensor,
                                     hetero_idx: torch.Tensor,
                                     model: torch.nn.Module,
                                     kernel_size: int,
                                     batch_size: int,
                                     device: torch.device) -> torch.Tensor:
    """
    Heterogeneous transpose: per-voxel predicted kernel and dot with local window.
    """
    D, H, W = prod_image.shape
    pad = kernel_size // 2
    offsets = torch.arange(-pad, pad + 1, device=device)
    dz, dy, dx = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    dz, dy, dx = dz.flatten(), dy.flatten(), dx.flatten()
    dist = make_distance_grid(kernel_size, device=device, normalize="max", negate=True)

    out = torch.zeros_like(prod_image, device=device)
    if hetero_idx.numel() == 0:
        return out

    model.eval()
    for start in tqdm(range(0, hetero_idx.size(0), batch_size), desc="[hetero TRP] batches"):
        end = min(start + batch_size, hetero_idx.size(0))
        b_idx = hetero_idx[start:end]  # [B,3]
        z, y, x = b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]

        # Extract μ and activity windows (clamped to borders)
        z_b = (z[:, None] + dz).clamp(0, D - 1)
        y_b = (y[:, None] + dy).clamp(0, H - 1)
        x_b = (x[:, None] + dx).clamp(0, W - 1)

        mu_win  = mu_image[z_b, y_b, x_b].view(-1, 1, kernel_size, kernel_size, kernel_size)
        act_win = prod_image[z_b, y_b, x_b].view(-1, kernel_size ** 3)

        with autocast():
            k_pred = model(mu_win, dist).view(-1, kernel_size ** 3)
            k_pred = F.softmax(k_pred, dim=1)

        voxel_val = torch.sum(act_win * k_pred, dim=1)  # [B]
        out[z, y, x] = voxel_val
    return out


def simulate_pr_transpose(mu_path: str,
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

    # ----- Load μ and snap to canonical materials (with clamp-to-Lung)
    mu = load_mu(mu_path, dim, device)

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
        mat = snap_mu_to_materials(mu)
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

    # ----- Load the “input” image for transpose (e.g., y or residual)
    prod_np = np.fromfile(act_path, np.float32).reshape(dim)
    prod = torch.from_numpy(prod_np).to(device)

    # ----- Homogeneous transpose (conv3d gather)
    out_hom = apply_kernels_convolution_transpose(prod, kernels, hom_map, (kernel_size,) * 3, device)

    # ----- Heterogeneous transpose (per-voxel predicted kernel)
    out_hetero = simulate_heterogeneous_transpose(mat, prod, hetero_idx, model, kernel_size, batch, device)

    out = out_hom + out_hetero

    # ----- Save
    #save_raw_3d(out_path, out.detach().cpu().numpy())
    return out #float(out.sum())


def build_argparser():
    p = argparse.ArgumentParser(description="Hybrid PR TRANSPOSE operator with caching (conv+model).")
    p.add_argument("--mu", required=True, help="Path to segmented μ-map (float32 raw).")
    p.add_argument("--act", required=True, help="Path to input image for transpose (float32 raw).")
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
    out = simulate_pr_transpose(
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
    out.tofile("/home/youness/data/PR_Correction/Reconstruction/SimPhantom/Recon_cnn/result_image_T.bin")
    print("Saved result_image to result_image.bin")

