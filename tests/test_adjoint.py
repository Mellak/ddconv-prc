"""
Dot-product (adjoint) test for the DDConv positron-range operators.

Checks <B(mu) x, y> == <x, B(mu)^T y> for random x and y, where B(mu) and
B(mu)^T are the functions CASToR calls (`simulate_pr_forward` in
inference/PR_CNN_gate.py and `simulate_pr_transpose` in inference/PR_CNN_gate_T.py).
Inputs are passed through temporary raw files, exactly as in the CASToR plugin.

The mu-map is large enough to contain homogeneous regions (convolution path)
and heterogeneous interfaces (per-voxel CNN-predicted PSFs), so both code paths
are exercised.

Run:  pytest -s tests/test_adjoint.py     (or: python tests/test_adjoint.py)
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "inference"))

from PR_CNN_gate import MATERIALS, simulate_pr_forward  # noqa: E402
from PR_CNN_gate_T import simulate_pr_transpose  # noqa: E402

KERNEL_SIZE = 11
DIM = (20, 24, 40)  # (Z, Y, X)
TOL = 1e-4


def make_mu(dim=DIM):
    mu = np.full(dim, MATERIALS["Water"], np.float32)
    mu[:, :, 22:] = MATERIALS["Lung"]
    mu[:, 8:12, 4:8] = MATERIALS["RibBone"]
    return mu


def run_operator(fn, mu_path, img, workdir, name):
    img_path = os.path.join(workdir, f"{name}.bin")
    img.astype(np.float32).tofile(img_path)
    out = fn(
        mu_path=mu_path, act_path=img_path, dim=img.shape,
        out_path=os.path.join(workdir, f"{name}_out.bin"),
        weights=os.path.join(ROOT, "inference", "Weights", f"fpred_model_{KERNEL_SIZE}.pth"),
        kernel_size=KERNEL_SIZE, kernel_dir=os.path.join(workdir, "kernels"),
        cache_dir=os.path.join(workdir, "cache"), batch=256, device_str="cpu",
    )
    return out.cpu().numpy().astype(np.float64)


def adjoint_relative_error(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(DIM, dtype=np.float32)
    y = rng.random(DIM, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        mu_path = os.path.join(tmp, "mu.bin")
        make_mu().tofile(mu_path)
        Bx = run_operator(simulate_pr_forward, mu_path, x, tmp, "x")
        BTy = run_operator(simulate_pr_transpose, mu_path, y, tmp, "y")
    lhs = float(np.dot(Bx.ravel(), y.astype(np.float64).ravel()))
    rhs = float(np.dot(x.astype(np.float64).ravel(), BTy.ravel()))
    return lhs, rhs, abs(lhs - rhs) / max(abs(lhs), abs(rhs))


def test_adjoint():
    lhs, rhs, rel = adjoint_relative_error()
    print(f"\n<Bx, y> = {lhs:.8e}\n<x, B^T y> = {rhs:.8e}\nrelative error = {rel:.3e}")
    assert rel < TOL


if __name__ == "__main__":
    test_adjoint()
