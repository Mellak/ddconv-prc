# ddconv-prc
Code for Dual-Input Dynamic Convolution (DDConv) for PET positron range correction

This repository accompanies the paper:

**Dual-Input Dynamic Convolution for Positron Range Correction in PET Image Reconstruction**  
Youness Mellak, Alexandre Bousse, Thibaut Merlin, Elise Émond, Mikko Hakulinen, Dimitris Visvikis  
arXiv:2503.00587

## Overview

This work introduces **Dual-Input Dynamic Convolution (DDConv)**, a learning-based framework for
positron range correction (PRC) in PET image reconstruction.  
DDConv dynamically predicts voxel-wise positron range point spread functions from local attenuation
information and integrates consistently into iterative reconstruction by providing both the forward
and transposed operators.

The method achieves near Monte-Carlo accuracy while remaining computationally efficient and
scanner-independent.

## Code availability

🚧 **Code will be released soon.**  
This repository is intentionally left empty for now and will be populated with:

- Training code for the DDConv PSF predictor
- Inference code for forward and transposed PR operators
- Example scripts for integration into iterative PET reconstruction
- Documentation and usage examples

## Citation

If you use this work, please cite:

```bibtex
@article{mellak2025ddconv,
  title={Dual-Input Dynamic Convolution for Positron Range Correction in PET Image Reconstruction},
  author={Mellak, Youness and Bousse, Alexandre and Merlin, Thibaut and Émond, Elise and Hakulinen, Mikko and Visvikis, Dimitris},
  journal={arXiv preprint arXiv:2503.00587},
  year={2025}
}
