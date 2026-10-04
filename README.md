# DDConv: Dual-Input Dynamic Convolution for Positron Range Correction

[![Paper](https://img.shields.io/badge/IEEE%20TRPMS-10.1109%2FTRPMS.2025.3647264-blue)](https://doi.org/10.1109/TRPMS.2025.3647264)
[![arXiv](https://img.shields.io/badge/arXiv-2503.00587-b31b1b)](https://arxiv.org/abs/2503.00587)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code for **"Dual-Input Dynamic Convolution for Positron Range Correction in PET Image
Reconstruction"**, IEEE TRPMS, 2025.

![DDConv](docs/figures/full_architecture.png)

## Overview

DDConv models positron range (PR) blurring for Ga-68 PET.
A CNN uses the local attenuation map μ and a distance map to predict one PSF for each voxel.
DDConv gives the forward operator B(μ) and its exact transpose B(μ)ᵀ.
You can use both operators in iterative reconstruction (ML-EM, OSEM) with the system model
H = A(μ) P B(μ).

DDConv continues our NSS/MIC 2024 work, which used a GNN to predict the weights of the PR operator
([DOI](https://doi.org/10.1109/NSS/MIC/RTSD57108.2024.10657784)). DDConv replaces the GNN with a
convolutional hypernetwork and dynamic convolution. See also our GAN positron path simulator,
[Particles_Tracking_GAN](https://github.com/Mellak/Particles_Tracking_GAN).

## Installation

```bash
git clone https://github.com/Mellak/ddconv-prc.git
cd ddconv-prc
pip install -r requirements.txt
```

For GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org).
For the full pipeline, you also need [GATE](http://www.opengatecollaboration.org/) and
[CASToR](https://castor-project.org/).

## Quick start

Run the pretrained model on CPU with a synthetic lung / water / bone phantom:

```bash
python examples/run_synthetic_phantom.py
```

The script writes `examples/output/synthetic_phantom_before_after.png`.

Make sure that B(μ)ᵀ is the adjoint of B(μ):

```bash
pytest -s tests/test_adjoint.py
```

The test checks ⟨Bx, y⟩ = ⟨x, Bᵀy⟩. The relative error is approximately 1e-6.

## Pretrained weights

| File | Kernel | Voxel size | Isotope |
|---|---|---|---|
| `inference/Weights/fpred_model_11.pth` | 11³ | 2 mm | Ga-68 |
| `inference/Weights/fpred_model_21.pth` | 21³ | 2 mm | Ga-68 |
| `inference/Weights/fpred_model_31.pth` | 31³ | 2 mm | Ga-68 |

We recommend the 11³ kernel. It gives the best balance between accuracy and speed.

## Full pipeline

1. **Generate the phantoms.** Make random 3D material phantoms:

   ```bash
   python phantoms/GeneratePhantoms.py 2.0 0001 --output_root <DATA>
   ```

2. **Simulate the PSFs with GATE.** Put a Ga-68 point source at the center of each phantom.
   See [`gate/ReadMe.md`](gate/ReadMe.md).

   ```bash
   Gate -a [resolution,2][number,0001] gate/main_PR.mac
   ```

3. **Prepare the dataset.** Crop the GATE outputs to 31³, 21³ and 11³:

   ```bash
   python phantoms/PrepareDataSet.py 0001 2.0 --data_root <DATA>
   ```

4. **Train the model.**

   ```bash
   python training/train.py --model filter --kernel_size 11 --loss kl \
       --epochs 5000 --batch_size 4 --lr 1e-4 --data_root <DATASET>
   ```

5. **Reconstruct with CASToR.** `inference/PR_CNN_gate.py` applies B(μ) and
   `inference/PR_CNN_gate_T.py` applies B(μ)ᵀ. Set your paths in `inference/mmatkernels.conf` and in
   the `__main__` block of the two scripts. See [`inference/ReadMe.md`](inference/ReadMe.md).

The `*.sh` files are SLURM scripts from our cluster. Change the paths before you use them.

## Results

DDConv PSFs agree with GATE Monte Carlo at lung, water and bone interfaces.
The SVTD baseline does not agree with GATE at these interfaces.

![Kernels](docs/figures/kernels_gate_svtd_ddconv.png)

XCAT reconstructions (MC-simulated data) over EM iterations:

![XCAT](docs/figures/xcat_reconstructions.png)

## Limitations

- The weights are for Ga-68 and 2-mm voxels only. For a different isotope or voxel size, simulate
  new data and train again.
- The model knows three materials only: lung, water and rib bone.
- The 11³ kernel keeps approximately 84% of the PSF energy in lung.

## Repository structure

```
phantoms/    Phantom generation and dataset preparation
gate/        GATE macros for Ga-68 point-source simulations
training/    Model and training script
inference/   B(μ) and B(μ)ᵀ operators for CASToR, pretrained weights
examples/    CPU demo
tests/       Adjoint test
docs/        Figures
```

## Citation

```bibtex
@article{mellak2025ddconv,
  title   = {Dual-Input Dynamic Convolution for Positron Range Correction in PET Image Reconstruction},
  author  = {Mellak, Youness and Bousse, Alexandre and Merlin, Thibaut and {\'E}mond, {\'E}lise and
             Hakulinen, Mikko and Visvikis, Dimitris},
  journal = {IEEE Transactions on Radiation and Plasma Medical Sciences},
  year    = {2025},
  doi     = {10.1109/TRPMS.2025.3647264}
}
```

## License

This project is under the [MIT License](LICENSE).
