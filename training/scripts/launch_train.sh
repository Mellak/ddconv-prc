#!/bin/bash
#SBATCH -p GPU11Go,GPU24Go,GPU48Go
#SBATCH --job-name=PR_PSF_Train
#SBATCH -N 1
##SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --error=OutErrFolder/train.err
#SBATCH --output=OutErrFolder/train.out
##SBATCH --time=2-00:00:00

# --- Ensure err/out folder exists ---
mkdir -p OutErrFolder

# === CONFIG ===
SIF="/homes/ymellak/python/SIF/pytorch2_pyg.sif"
PY="/homes/ymellak/PR_Correction/New_Training/PythonCode/train.py"

# --- Main hyperparameters you edit ---
MODEL="filter"           # <- CHOOSE: "unet" (previous) or "fastunet" or filterpredictor (proposed fast model)
KERNEL_SIZE=25         # 11,13, 21,25 or 31,37
BASE_CH=16              # base channels (used by both models)
LOSS="kl"              # l1, l2, huber, kl
EPOCHS=5000
BATCH=4
TBATCH=2
LR=1e-4
WD=0.0
NUM_WORKERS=0

# Data selection (leave empty to auto-select by kernel_size)
MAX_SAMPLES=           # e.g., 2000 or empty
DATA_ROOT=             # e.g., /homes/ymellak/... or empty to auto-pick

# File extensions
EMIT_EXT=".raw"
MUMAP_EXT=".bin"
STOP_EXT=".raw"

# Dataset normalization / spike
NORM_PROD=1
NORM_STOP=1
NORM_MODE="sum"
EPS=1e-12
SPIKE_CENTER=1
SPIKE_VALUE=1
USE_MEMMAP=0

# Logging
LOG_INTERVAL=100
HUBER_DELTA=1.0

# Distance-channel (must match train.py)
NEGATE_DIST=1                 # 1 = use negative distances (recommended)
DIST_NORMALIZE="max"          # "max" or "none" (recommended: max)

# --- Auto-generate run name (now includes model) ---
RUN_NAME="${MODEL}_k${KERNEL_SIZE}_${LOSS}_b${BASE_CH}_dist-${DIST_NORMALIZE}_neg-${NEGATE_DIST}"
echo "[INFO] RUN_NAME=$RUN_NAME"

# --- Build argument list ---
ARGS=()
ARGS+=(--model "$MODEL")               # << NEW: pass model choice to train.py
ARGS+=(--kernel_size "$KERNEL_SIZE")
ARGS+=(--epochs "$EPOCHS")
ARGS+=(--batch_size "$BATCH")
ARGS+=(--test_batch_size "$TBATCH")
ARGS+=(--lr "$LR")
ARGS+=(--weight_decay "$WD")
ARGS+=(--base_channels "$BASE_CH")
ARGS+=(--loss "$LOSS")
ARGS+=(--huber_delta "$HUBER_DELTA")
ARGS+=(--num_workers "$NUM_WORKERS")
ARGS+=(--run_name "$RUN_NAME")
ARGS+=(--emission_ext "$EMIT_EXT")
ARGS+=(--mumap_ext "$MUMAP_EXT")
ARGS+=(--stop_ext "$STOP_EXT")
ARGS+=(--normalize_prod "$NORM_PROD")
ARGS+=(--normalize_stop "$NORM_STOP")
ARGS+=(--norm_mode "$NORM_MODE")
ARGS+=(--eps "$EPS")
ARGS+=(--spike_center "$SPIKE_CENTER")
ARGS+=(--spike_value "$SPIKE_VALUE")
ARGS+=(--use_memmap "$USE_MEMMAP")
ARGS+=(--log_interval "$LOG_INTERVAL")
# Distance flags
ARGS+=(--negate_dist "$NEGATE_DIST")
ARGS+=(--dist_normalize "$DIST_NORMALIZE")

# Optional args only if non-empty
if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi
if [[ -n "$DATA_ROOT" ]]; then
  ARGS+=(--data_root "$DATA_ROOT")
fi

# --- Launch ---
echo "[RUN] Launching training with: model=$MODEL, kernel_size=$KERNEL_SIZE, loss=$LOSS, base_ch=$BASE_CH, dist_norm=$DIST_NORMALIZE, negate=$NEGATE_DIST"
echo "[RUN] Using RUN_NAME=$RUN_NAME"
srun singularity run --nv "$SIF" python "$PY" "${ARGS[@]}"

