# Inference (CASToR plugin) — DDConv Positron Range Correction (Ga-68)

This folder contains the **inference-side implementation** of DDConv for **positron range correction (PRC)**, designed to be **used directly inside CASToR** as part of the reconstruction forward / backward model. :contentReference[oaicite:0]{index=0}

## Folder content
- `Weights/`  
  Example pretrained weights (you can replace with your own).

- `model.py`  
  Neural network definition used at inference time.  
  We use **FilterPredictor**, as it provides a good equilibrium between **accuracy and latency** for iterative reconstruction loops.

- `PR_CNN_gate.py`  
  Implements the **positron range operator** \( B(\mu) \): it applies **PR blurring** using voxel-wise kernels predicted by the model.

- `PR_CNN_gate_T.py`  
  Implements the **transpose** \( B(\mu)^{\top} \) of the positron range operator (the adjoint required for iterative reconstruction).

- `utils.py`  
  Helpers (I/O, reshaping, normalization utilities, etc.).

- `mmatkernels.conf`  
  CASToR configuration file to plug this operator into the reconstruction chain.

## How it plugs into CASToR (operator view)
In the paper notation, the PET system matrix is decomposed as: :contentReference[oaicite:1]{index=1}

\[
H = A(\mu)\, P\, B(\mu)
\]

where:
- \(P\) is the geometric projector,
- \(A(\mu)\) accounts for attenuation along LORs,
- \(B(\mu)\) is the **positron range blurring operator**.

Here:
- `PR_CNN_gate.py` implements **\( B(\mu) \)** (positron range operator),
- `PR_CNN_gate_T.py` implements **\( B(\mu)^{\top} \)** (its transpose). :contentReference[oaicite:2]{index=2}

This is exactly what is needed to keep a **matched forward/backward model** inside EM reconstruction.

## MLEM update (paper notation)
The MLEM / EM update rule (paper Eq. (3)) is: :contentReference[oaicite:3]{index=3}

\[
x^{(q+1)} =
\frac{x^{(q)}}{H^{\top}\mathbf{1}}
\; H^{\top}\!\left(\frac{y}{H x^{(q)} + r}\right)
\]

with element-wise operations, where:
- \(x\) is the activity image,
- \(y\) is the measured data,
- \(r\) is the background term (randoms/scatter),
- \(H\) is the forward model and \(H^{\top}\) its transpose.

Because \(H\) includes \(B(\mu)\), CASToR needs both:
- \(B(\mu)\) (forward PR operator) → `PR_CNN_gate.py`
- \(B(\mu)^{\top}\) (transpose PR operator) → `PR_CNN_gate_T.py` 

## Notes
- The provided weights are **example weights** for convenience.
- This implementation is currently provided for **Ga-68** (training data + learned kernels are tracer/voxel-size dependent).
