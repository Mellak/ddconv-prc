"""
PrepareDataSet.py

This script prepares the Positron Range simulation outputs into structured datasets
for deep learning tasks.

Given a simulated phantom and a target voxel resolution, it:
  1. Loads the full 3D emission, stop, and attenuation (μ-map) volumes
     produced by GATE.
  2. Infers the cubic kernel size directly from the raw files.
  3. Generates three dataset variants:
       - FULL   : the complete kernel
       - CROP   : a centered crop around the emission voxel
       - EXCROP : a smaller centered crop
     where crop sizes are scaled consistently with voxel resolution.
  4. Saves each variant into a standardized folder structure
     (Emission / Stop / MuMap) using raw binary formats.


Typical usage:
    python PrepareDataSet.py <phantom_id> <resolution_mm>

Example:
    python PrepareDataSet.py 1001 2.0
"""


import os
import argparse
import numpy as np
from typing import Tuple

# -------------------------------------------------
# Argument parser (ALL defaults defined here)
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Ga68 dataset: full / crop / excrop kernels"
    )

    # Required
    parser.add_argument("phantom_number", type=str,
                        help="Phantom index (e.g. 1001)")
    parser.add_argument("resolution_mm", type=float,
                        help="Voxel resolution in mm")

    # Dataset geometry
    parser.add_argument("--base_resolution_mm", type=float, default=2.0,
                        help="Reference resolution in mm (default: 2.0)")
    parser.add_argument("--base_sizes", type=int, nargs=3,
                        default=(31, 21, 11),
                        metavar=("FULL", "CROP", "EXCROP"),
                        help="Kernel sizes at base resolution (default: 31 21 11)")

    # I/O
    parser.add_argument("--data_root", type=str,
                        default="/homes/ymellak/PR_Correction/NewData",
                        help="Root directory for all data")

    # Data type
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Numpy dtype for raw/bin files (default: float32)")

    return parser.parse_args()


# -------------------------------------------------
# Utilities (UNCHANGED)
# -------------------------------------------------

def round_to_odd(x: float) -> int:
    n = int(round(x))
    if n <= 0:
        return 1
    if n % 2 == 1:
        return n
    up, dn = n + 1, n - 1
    return up if abs(up - x) <= abs(dn - x) else max(1, dn)


def scaled_sizes(target_res_mm: float,
                 base_res_mm: float,
                 base_sizes: Tuple[int, int, int]) -> Tuple[int, int, int]:
    phys_mm = [s * base_res_mm for s in base_sizes]
    return tuple(round_to_odd(p / target_res_mm) for p in phys_mm)


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def infer_cubic_edge_from_file(path: str, dtype: np.dtype) -> int:
    nbytes = os.path.getsize(path)
    itemsize = np.dtype(dtype).itemsize
    if nbytes % itemsize != 0:
        raise ValueError(f"{path}: file size not divisible by dtype size.")
    count = nbytes // itemsize
    N = int(round(count ** (1 / 3)))
    if N ** 3 != count:
        raise ValueError(f"{path}: data is not a cubic volume.")
    return N


def read_raw_cubic(path: str, N: int, dtype: np.dtype) -> np.ndarray:
    arr = np.fromfile(path, dtype=dtype)
    if arr.size != N ** 3:
        raise ValueError(f"{path}: wrong element count.")
    return arr.reshape((N, N, N))


def write_raw(path: str, arr: np.ndarray):
    arr.astype(arr.dtype, copy=False).tofile(path)


def center_crop_cubic(vol: np.ndarray, out_N: int) -> np.ndarray:
    N = vol.shape[0]
    r = out_N // 2
    c = N // 2
    return vol[c - r:c + r + 1,
               c - r:c + r + 1,
               c - r:c + r + 1]


# -------------------------------------------------
# Core logic (UNCHANGED behavior)
# -------------------------------------------------

def copy_full_and_make_crops(phantom_number: str,
                             resolution_mm: float,
                             dtype: np.dtype,
                             base_resolution_mm: float,
                             base_sizes: Tuple[int, int, int],
                             data_root: str):

    full_N_exp, crop_N, excrop_N = scaled_sizes(
        resolution_mm, base_resolution_mm, base_sizes
    )

    src_root = f"{data_root}/Phantoms_Ga68_{resolution_mm}mm"
    ph_dir = os.path.join(src_root, f"Phantom_{phantom_number}")

    prod_src = os.path.join(ph_dir, f"Prod_{phantom_number}-Prod.raw")
    stop_src = os.path.join(ph_dir, f"Prod_{phantom_number}-Stop.raw")
    mu_src   = os.path.join(ph_dir, "mu_map511-MuMap.bin")

    for p in (prod_src, stop_src, mu_src):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    N = infer_cubic_edge_from_file(prod_src, dtype)

    emission = read_raw_cubic(prod_src, N, dtype)
    stop     = read_raw_cubic(stop_src, N, dtype)
    mumap    = read_raw_cubic(mu_src, N, dtype)

    full_root   = f"{data_root}/Dataset_{resolution_mm}mm"
    crop_root   = f"{data_root}/Dataset_crop_{resolution_mm}mm"
    excrop_root = f"{data_root}/Dataset_Excrop_{resolution_mm}mm"

    def trio(root):
        return (os.path.join(root, "Emission"),
                os.path.join(root, "Stop"),
                os.path.join(root, "MuMap"))

    ensure_dirs(*trio(full_root), *trio(crop_root), *trio(excrop_root))

    # FULL
    write_raw(f"{full_root}/Emission/Emission_{phantom_number}.raw", emission)
    write_raw(f"{full_root}/Stop/Stop_{phantom_number}.raw", stop)
    write_raw(f"{full_root}/MuMap/MuMap_{phantom_number}.bin", mumap)

    # CROP
    write_raw(f"{crop_root}/Emission/Emission_{phantom_number}.raw",
              center_crop_cubic(emission, crop_N))
    write_raw(f"{crop_root}/Stop/Stop_{phantom_number}.raw",
              center_crop_cubic(stop, crop_N))
    write_raw(f"{crop_root}/MuMap/MuMap_{phantom_number}.bin",
              center_crop_cubic(mumap, crop_N))

    # EXCROP
    write_raw(f"{excrop_root}/Emission/Emission_{phantom_number}.raw",
              center_crop_cubic(emission, excrop_N))
    write_raw(f"{excrop_root}/Stop/Stop_{phantom_number}.raw",
              center_crop_cubic(stop, excrop_N))
    write_raw(f"{excrop_root}/MuMap/MuMap_{phantom_number}.bin",
              center_crop_cubic(mumap, excrop_N))

    print(f"[OK] Phantom {phantom_number} | res={resolution_mm}mm")


# -------------------------------------------------
# Entry point
# -------------------------------------------------

def main():
    args = parse_args()

    try:
        dtype = np.dtype(args.dtype)
    except Exception:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    copy_full_and_make_crops(
        phantom_number=args.phantom_number,
        resolution_mm=args.resolution_mm,
        dtype=dtype,
        base_resolution_mm=args.base_resolution_mm,
        base_sizes=tuple(args.base_sizes),
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
