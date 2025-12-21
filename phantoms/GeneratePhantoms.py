#!/usr/bin/env python3
"""
GeneratePhantomsGa68.py

This script generates synthetic 3D voxelized phantoms for Positron Range simulations.

For a reference resolution of 2.0 mm, the generated volume has a size of 31×31×31
voxels. This size is derived from the maximum effective range of Ga-68 positrons
in lung tissue, which is approximately 30 mm. The volume is constructed to fully
cover this physical extent in all directions, ensuring complete positron range
coverage.

The phantom generation process includes:
  - A heterogeneous background composed of random material zones.
  - Random geometric structures (boxes, spheres, cylinders, pyramids, rectangles)
    placed near the center of the volume to mimic complex tissue distributions.
  - Controlled random voxel perturbations to introduce variability.

The volume size automatically scales with voxel resolution to preserve the same
physical coverage. If a different radioisotope is used, the maximum positron range
must be updated accordingly (via the --max_dist_of_radius parameter) to ensure
proper spatial coverage.

All geometry, material, noise, and output parameters are configurable via
command-line arguments, with defaults corresponding to the Ga-68, 2 mm reference
setup.

Typical usage:
    python GeneratePhantoms.py <resolution_mm> <phantom_id>

Example:
    python GeneratePhantoms.py 2.0 001
"""


import numpy as np
import os
import random
import argparse

# -------------------------------------------------
# Argument parser (defaults = original hard-coded values)
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic 3D Ga68 phantoms")

    # Required (previously sys.argv)
    parser.add_argument("resolution", type=float, help="Voxel size in mm")
    parser.add_argument("image_idx", type=str, help="Phantom index")

    # Geometry / scaling -- If changing isotope, update max_dist_of_radius and base_resolution
    parser.add_argument("--max_dist_of_radius", type=float, default=30.0,
                        help="Max radius distance in mm (default: 30 for Ga68 @ 2mm)")
    parser.add_argument("--base_resolution", type=float, default=2.0,
                        help="Reference resolution in mm (default: 2.0)")

    # Materials
    parser.add_argument("--num_materials", type=int, default=10,
                        help="Number of material labels (default: 10)")

    # Random zones
    parser.add_argument("--min_zones", type=int, default=5,
                        help="Minimum number of background zones (default: 5)")
    parser.add_argument("--max_zones", type=int, default=10,
                        help="Maximum number of background zones (default: 10)")
    parser.add_argument("--min_zone_frac", type=float, default=0.05,
                        help="Minimum zone thickness fraction (default: 0.05)")
    parser.add_argument("--max_zone_frac", type=float, default=0.30,
                        help="Maximum zone thickness fraction (default: 0.30)")

    # Shapes
    parser.add_argument("--num_shapes", type=int, default=12,
                        help="Number of random shapes (default: 12)")
    parser.add_argument("--shape_size_min", type=int, default=2,
                        help="Min shape size before scaling (default: 2)")
    parser.add_argument("--shape_size_max", type=int, default=10,
                        help="Max shape size before scaling (default: 10)")
    parser.add_argument("--offset_base", type=int, default=5,
                        help="Base offset range before scaling (default: 5)")

    # Noise
    parser.add_argument("--random_voxel_fraction", type=float, default=0.10,
                        help="Fraction of voxels randomly modified (default: 0.10)")

    # Output
    parser.add_argument("--output_root", type=str,
                        default="/homes/ymellak/PR_Correction/NewData",
                        help="Root output directory")

    return parser.parse_args()


# -------------------------------------------------
# Shape / fill functions (UNCHANGED behavior)
# -------------------------------------------------

def fill_with_random_zones(image, materials, orientation, min_zones, max_zones,
                           min_zone_frac, max_zone_frac):
    size = image.shape[0]
    n_zones = random.randint(min_zones, max_zones)
    min_thick = int(min_zone_frac * size)
    max_thick = int(max_zone_frac * size)

    thicknesses = []
    remaining = size
    for i in range(n_zones):
        if i == n_zones - 1:
            thick = remaining
        else:
            max_for_this = min(max_thick, remaining - min_thick * (n_zones - i - 1))
            thick = random.randint(min_thick, max_for_this)
        thicknesses.append(thick)
        remaining -= thick

    axis = {"vertical": 1, "horizontal": 0}[orientation]
    pos = 0
    for thick in thicknesses:
        material = random.choice(materials)
        if axis == 0:
            image[pos:pos + thick, :, :] = material
        else:
            image[:, pos:pos + thick, :] = material
        pos += thick


def generate_box(image, center, size, window_size, materials):
    material = np.random.choice(materials)
    center = np.array(center)
    half = size // 2
    start = np.maximum(center - half, 0)
    end = np.minimum(center + half + 1, window_size)
    image[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = material


def generate_sphere(image, center, radius, window_size, materials):
    material = np.random.choice(materials)
    center = np.array(center)
    for x in range(center[0] - radius, center[0] + radius + 1):
        for y in range(center[1] - radius, center[1] + radius + 1):
            for z in range(center[2] - radius, center[2] + radius + 1):
                if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                    if 0 <= x < window_size and 0 <= y < window_size and 0 <= z < window_size:
                        image[x, y, z] = material


def generate_pyramid(image, center, height, window_size, materials):
    material = np.random.choice(materials)
    center = np.array(center)
    for i in range(height):
        size = height - i
        generate_box(image, center + [0, 0, i], size, window_size, [material])


def generate_cylinder(image, center, radius, height, window_size, materials):
    material = np.random.choice(materials)
    center = np.array(center)
    for z in range(center[2] - height // 2, center[2] + height // 2 + 1):
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                    if 0 <= x < window_size and 0 <= y < window_size and 0 <= z < window_size:
                        image[x, y, z] = material


def generate_rectangle(image, center, sx, sy, sz, window_size, materials):
    material = np.random.choice(materials)
    center = np.array(center)
    start = np.maximum(center - [sx // 2, sy // 2, sz // 2], 0)
    end = np.minimum(center + [sx // 2, sy // 2, sz // 2] + 1, window_size)
    image[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = material


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    args = parse_args()

    resolution = args.resolution
    voxel_size = resolution
    scale_factor = args.base_resolution / voxel_size

    window_size = int(args.max_dist_of_radius / voxel_size * 2)
    if window_size % 2 == 0:
        window_size += 1

    materials = list(range(1, args.num_materials + 1))

    # Initialize image
    image = np.ones((window_size, window_size, window_size), dtype=np.float32)
    image *= np.random.choice(materials)

    # Background zones
    orientation = random.choice(["vertical", "horizontal"])
    fill_with_random_zones(
        image, materials, orientation,
        args.min_zones, args.max_zones,
        args.min_zone_frac, args.max_zone_frac
    )

    central = (window_size - 1) // 2

    # Shapes
    for _ in range(args.num_shapes):
        shape = random.choice(["box", "sphere", "pyramid", "cylinder", "rectangle"])
        size = max(1, int(random.randint(args.shape_size_min,
                                         args.shape_size_max) * scale_factor))
        offset_range = int(args.offset_base * scale_factor)
        offset = np.random.randint(-offset_range, offset_range + 1, size=3)
        center = central + offset

        if shape == "box":
            generate_box(image, center, size, window_size, materials)
        elif shape == "sphere":
            generate_sphere(image, center, size // 2, window_size, materials)
        elif shape == "pyramid":
            generate_pyramid(image, center, size, window_size, materials)
        elif shape == "cylinder":
            generate_cylinder(image, center, size // 2, size, window_size, materials)
        elif shape == "rectangle":
            generate_rectangle(image, center, size, size, size // 2, window_size, materials)

    # Random voxel modification (10%)
    num_modify = int(image.size * args.random_voxel_fraction)
    idx = np.random.choice(image.size, num_modify, replace=False)
    flat = image.ravel()
    flat[idx] = np.random.choice(materials, size=num_modify)
    image = flat.reshape(image.shape)

    # Output
    base_path = (
        f"{args.output_root}/Phantoms_Ga68_{resolution}mm/"
        f"Phantom_{args.image_idx}"
    )
    os.makedirs(base_path, exist_ok=True)

    bin_path = f"{base_path}/image_{args.image_idx}.bin"
    image.astype(np.float32).tofile(bin_path)

    h33_path = f"{base_path}/image_{args.image_idx}.h33"
    with open(h33_path, "w") as f:
        f.write("!INTERFILE  :=\n")
        f.write(f"!matrix size [1] := {window_size}\n")
        f.write(f"!matrix size [2] := {window_size}\n")
        f.write(f"!name of data file := {bin_path}\n")
        f.write("!number format := short float\n")
        f.write("imagedata byte order := LITTLEENDIAN\n")
        f.write(f"scaling factor (mm/pixel) [1] := {voxel_size}\n")
        f.write(f"scaling factor (mm/pixel) [2] := {voxel_size}\n")
        f.write(f"!number of slices := {window_size}\n")
        f.write(f"slice thickness (pixels) := {voxel_size}\n")

    print(f"Generated .h33 file at {h33_path}")


if __name__ == "__main__":
    main()
