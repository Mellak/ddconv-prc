#!/usr/bin/env python3
"""
Run the included DDConv weights on CPU on a small synthetic Ga-68 phantom.

The phantom (2-mm voxels) contains water, a lung block and a rib-bone slab, with
hot lesions inside the lung, at the lung/water interface and next to the bone.
The script applies the positron-range blurring operator B(mu) implemented in
`inference/PR_CNN_gate.py` (the same function CASToR calls) and saves a
before/after figure.

Usage:
    python examples/run_synthetic_phantom.py [--out examples/output] [--kernel-size 11]
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "inference"))

from PR_CNN_gate import MATERIALS, simulate_pr_forward  # noqa: E402

DIM = (24, 48, 48)  # (Z, Y, X), 2-mm voxels


def make_phantom(dim=DIM):
    """Return (mu, activity) float32 arrays of shape dim."""
    Z, Y, X = dim
    mu = np.full(dim, MATERIALS["Water"], np.float32)
    mu[:, 10:38, 24:44] = MATERIALS["Lung"]       # lung block
    mu[:, 6:42, 8:12] = MATERIALS["RibBone"]      # bone slab

    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    act = np.full(dim, 1.0, np.float32)           # warm background
    act[mu == MATERIALS["Lung"]] = 0.2            # low uptake in lung
    lesions = [  # (z, y, x, radius in voxels)
        (12, 18, 34, 3),  # inside lung
        (12, 30, 24, 3),  # at the lung/water interface
        (12, 24, 14, 2),  # next to the bone slab
    ]
    for z, y, x, r in lesions:
        act[(zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2] = 10.0
    return mu, act


def apply_forward(mu, act, weights, kernel_size, workdir, batch=256):
    mu_path = os.path.join(workdir, "mu.bin")
    act_path = os.path.join(workdir, "act.bin")
    mu.astype(np.float32).tofile(mu_path)
    act.astype(np.float32).tofile(act_path)
    out = simulate_pr_forward(
        mu_path=mu_path, act_path=act_path, dim=mu.shape,
        out_path=os.path.join(workdir, "out.bin"), weights=weights,
        kernel_size=kernel_size, kernel_dir=os.path.join(workdir, "kernels"),
        cache_dir=os.path.join(workdir, "cache"), batch=batch, device_str="cpu",
    )
    return out.cpu().numpy()


def save_figure(mu, act, blurred, path, z=12, y=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    mat = np.select([mu[z] == MATERIALS["Lung"], mu[z] == MATERIALS["Water"]], [0, 1], 2)
    ax[0].imshow(mat, cmap=ListedColormap(["pink", "lightblue", "gray"]), vmin=0, vmax=2)
    ax[0].set_title("Materials (lung / water / bone)")
    vmax = act.max()
    ax[1].imshow(act[z], cmap="gray_r", vmin=0, vmax=vmax)
    ax[1].set_title("Activity x")
    ax[2].imshow(blurred[z], cmap="gray_r", vmin=0, vmax=vmax)
    ax[2].set_title(r"DDConv blurred $B(\mu)\,x$")
    for a in ax[:3]:
        a.axhline(y, color="tab:orange", lw=0.8, ls="--")
        a.set_xticks([]); a.set_yticks([])
    xs = np.arange(mu.shape[2]) * 2.0
    ax[3].plot(xs, act[z, y], label="activity", color="black")
    ax[3].plot(xs, blurred[z, y], label="blurred (DDConv)", color="tab:orange")
    ax[3].set_xlabel("x [mm]"); ax[3].set_title("Profile (dashed line)")
    ax[3].legend(frameon=False)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default=os.path.join(ROOT, "examples", "output"))
    p.add_argument("--kernel-size", type=int, default=11, choices=[11, 21, 31])
    args = p.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    weights = os.path.join(ROOT, "inference", "Weights", f"fpred_model_{args.kernel_size}.pth")

    mu, act = make_phantom()
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        blurred = apply_forward(mu, act, weights, args.kernel_size, tmp)
        dt = time.time() - t0

    fig_path = os.path.join(args.out, "synthetic_phantom_before_after.png")
    save_figure(mu, act, blurred, fig_path)
    print(f"grid {mu.shape}, kernel {args.kernel_size}^3, CPU time {dt:.1f} s")
    print(f"total activity before {act.sum():.1f}, after {blurred.sum():.1f} "
          f"(ratio {blurred.sum() / act.sum():.4f}; each PSF sums to 1)")
    print(f"peak activity before {act.max():.2f}, after {blurred.max():.2f}")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
