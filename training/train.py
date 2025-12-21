#!/usr/bin/env python3
# train.py
# Parametric training for voxel-wise PSF kernel prediction.
# - Uses FilterPredictor from model.py
# - Dataset: new DataSetManager2C (configurable normalization, spike, extensions, etc.)
# - Loss: L1 / L2 / Huber / KL (choose via --loss)
# - For L1/L2/Huber: matches your original pipeline:
#     prob = softmax(logits), pred_Y = prob * center_voxel, loss(pred_Y, target)
# - For KL: compares distributions:
#     prob = softmax(logits), target_prob = target / sum(target), KL(target||prob)
# - Outputs under /homes/ymellak/PR_Correction/New_Training/Expirements/<run_name>/

import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from model import FilterPredictorUNet, FastUNet3D, FilterPredictor            # your model (separate file)
from DataLoader import DataSetManager2C          # your dataset (new version)

# ---------------------------
# Helpers
# ---------------------------

def make_distance_grid(
    size: int,
    device: torch.device,
    dtype=torch.float32,
    negate: bool = True,
    normalize: str = "max",   # "max" -> dist/dist.max ; "none" -> raw voxel units
):
    """
    Euclidean distance-to-center grid for a cubic kernel (size^3).
    Returns: [1,1,size,size,size] tensor on device.

    Best practice for PR: use NEGATIVE, NORMALIZED distance.
      - negate=True  -> -dist
      - normalize="max" -> divide by dist.max so the range becomes [-1, 0]
    """
    c = size // 2
    z, y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype),
        torch.arange(size, device=device, dtype=dtype),
        indexing='ij'
    )
    dist = torch.sqrt((z - c)**2 + (y - c)**2 + (x - c)**2)

    if normalize.lower() == "max":
        dmax = dist.max().clamp_min(1e-8)
        dist = dist / dmax
    elif normalize.lower() == "none":
        pass
    else:
        raise ValueError(f"--dist_normalize must be 'max' or 'none', got '{normalize}'")

    if negate:
        dist = -dist

    return dist.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

def resolve_dataset_roots(
    kernel_size: int,
    # 2.0 mm datasets
    root11="/homes/ymellak/PR_Correction/Phantoms/Dataset_Excrop_2mm/",
    root21="/homes/ymellak/PR_Correction/Phantoms/Dataset_crop_2mm/",
    root31="/homes/ymellak/PR_Correction/Phantoms/Dataset_2mm/",
    # 1.65 mm datasets
    root13="/homes/ymellak/PR_Correction/NewData/Dataset_Excrop_1.65mm/",
    root25="/homes/ymellak/PR_Correction/NewData/Dataset_crop_1.65mm/",
    root37="/homes/ymellak/PR_Correction/NewData/Dataset_1.65mm/",
    override: str = None
):
    """
    Return (emission_dir, mumap_dir, stop_dir) based on kernel_size or override root.
    Supported kernel sizes:
        11 -> 2 mm extreme crop
        21 -> 2 mm crop
        31 -> 2 mm full
        13 -> 1.65 mm extreme crop
        25 -> 1.65 mm crop
        37 -> 1.65 mm full
    """
    if override is not None:
        base = override
    else:
        if   kernel_size == 11: base = root11
        elif kernel_size == 21: base = root21
        elif kernel_size == 31: base = root31
        elif kernel_size == 13: base = root13
        elif kernel_size == 25: base = root25
        elif kernel_size == 37: base = root37
        else:
            raise ValueError(
                f"Unsupported kernel_size={kernel_size}. "
                "Use one of: 11, 13, 21, 25, 31, 37 (or pass --data_root)."
            )

    return (
        os.path.join(base, "Emission"),
        os.path.join(base, "MuMap"),
        os.path.join(base, "Stop"),
    )

def ensure_dirs(base_out_dir: str, run_name: str):
    base = os.path.join(base_out_dir, run_name)
    paths = {
        "base": base,
        "weights": os.path.join(base, "weights"),
        "images": os.path.join(base, "images"),
        "logs": os.path.join(base, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def save_args(args, paths):
    cfg_path = os.path.join(paths["logs"], "args.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[cfg] saved args -> {cfg_path}")

def save_checkpoint(epoch, model, optimizer, best_loss, weights_dir):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    ckpt_path = os.path.join(weights_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, ckpt_path)
    print(f"[ckpt] saved {ckpt_path}")
    return ckpt_path

def maybe_load_best(model, optimizer, weights_dir, map_location=None):
    best_path = os.path.join(weights_dir, "best_model.pth")
    if not os.path.isfile(best_path):
        print("[ckpt] no best_model.pth (fresh run).")
        return 0, float("inf")
    state = torch.load(best_path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    best_loss = state.get("best_loss", float("inf"))
    print(f"[ckpt] loaded best model from epoch {state['epoch']} (best_loss={best_loss:.6f})")
    return state["epoch"], best_loss

def update_best_if_needed(current_loss, best_loss, epoch, model, optimizer, weights_dir):
    if current_loss < best_loss:
        best_loss = current_loss
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        best_path = os.path.join(weights_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"[ckpt] ★ updated best_model.pth at epoch {epoch} (loss={current_loss:.6f})")
    return best_loss

def create_dataloaders(emission_dir, mumap_dir, stop_dir,
                       kernel_size: int,
                       batch_size: int, test_batch_size: int,
                       num_workers: int,
                       max_samples: int,
                       ds_kwargs: dict):
    """
    Wrap DataSetManager2C -> train & test loaders (mirrors same set unless you split).
    """
    dataset = DataSetManager2C(
        emission_dir, mumap_dir, stop_dir,
        max_samples=max_samples,
        img_size=(kernel_size, kernel_size, kernel_size),
        **ds_kwargs
    )
    indices = list(range(len(dataset)))
    train_set = Subset(dataset, indices)
    test_set  = Subset(dataset, indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader

# ---------------------------
# Loss selection
# ---------------------------

def make_criterion(loss_name: str, huber_delta: float = 1.0):
    """
    Returns a tuple (criterion_fn, uses_distribution)
    - For 'kl': the criterion_fn expects (prob_pred, prob_target)
    - For others: criterion_fn expects (pred_Y, target_img)
    """
    loss = loss_name.lower()
    if loss in ["l1", "mae"]:
        crit = nn.L1Loss()
        return crit, False
    elif loss in ["l2", "mse"]:
        crit = nn.MSELoss()
        return crit, False
    elif loss in ["huber", "smoothl1", "smooth_l1"]:
        crit = nn.SmoothL1Loss(beta=huber_delta)
        return crit, False
    elif loss in ["kl", "kld", "kldiv", "kullback", "kullbackleibler"]:
        # Use manual KL: D_KL(target || pred)
        def kl_divergence(pred_prob, target_prob, eps=1e-8):
            p = target_prob.clamp_min(eps)
            q = pred_prob.clamp_min(eps)
            # sum over all voxels per item, then mean over batch
            return (p * (p.log() - q.log())).sum(dim=(1,2,3,4)).mean()
        return kl_divergence, True
    else:
        raise ValueError(f"Unsupported --loss '{loss_name}'. Choose from: l1, l2, huber, kl.")

# ---------------------------
# Train / Eval
# ---------------------------

def model_train_step(model, optimizer, criterion, uses_distribution,
                     input_img, mu_img, target_img,
                     distances, kernel_size):
    """
    - If uses_distribution False (L1/L2/Huber): original pipeline.
    - If uses_distribution True (KL): compare distributions (sum=1).
    """
    optimizer.zero_grad(set_to_none=True)

    logits = model(mu_img, distances)  # [B,1,N,N,N]
    prob   = F.softmax(logits.reshape(logits.size(0), -1), dim=1)\
               .reshape(-1, 1, kernel_size, kernel_size, kernel_size)

    if not uses_distribution:
        center_voxel = torch.sum(input_img, dim=(1, 2, 3, 4), keepdim=True)
        pred_Y = prob * center_voxel
        loss = criterion(pred_Y, target_img)
    else:
        # KL on distributions: normalize target to prob. space
        target_sum = target_img.sum(dim=(1,2,3,4), keepdim=True).clamp_min(1e-8)
        target_prob = target_img / target_sum
        loss = criterion(prob, target_prob)

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def model_predict(model, input_img, mu_img, distances, kernel_size, uses_distribution):
    logits = model(mu_img, distances)
    prob   = F.softmax(logits.reshape(logits.size(0), -1), dim=1)\
               .reshape(-1, 1, kernel_size, kernel_size, kernel_size)
    if uses_distribution:
        # For KL mode, return the probability PSF (sum=1)
        return prob
    else:
        center_voxel = torch.sum(input_img, dim=(1, 2, 3, 4), keepdim=True)
        pred_Y = prob * center_voxel
        return pred_Y

def save_mid_slice_png(save_dir, epoch, input_img, mu_img, target_img, pred_img, uses_distribution, vmax=10000.0):
    """
    Saves a mid-slice panel (XY) for the first item in the batch.
    If KL: pred_img is prob; display with a robust vmax.
    """
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)

    b = 0
    N = input_img.size(2)
    mid = N // 2

    inp = input_img[b, 0, mid].detach().cpu().numpy()
    mu  = mu_img[b, 0, mid].detach().cpu().numpy()
    tgt = target_img[b, 0, mid].detach().cpu().numpy()
    pred = pred_img[b, 0, mid].detach().cpu().numpy()

    if uses_distribution:
        vmax_use = pred.max() if pred.max() > 0 else 1.0
        title_pred = "Pred Prob"
        title_tgt  = "Target (img)"  # still raw target
        s = float(target_img[b].sum().item())
        diff = tgt - pred * s
    else:
        vmax_use = vmax
        title_pred = "Prediction"
        title_tgt  = "Target"
        diff = tgt - pred

    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))
    ax[0].imshow(inp, cmap="gray");                ax[0].set_title("Input")
    ax[1].imshow(mu, cmap="gray");                 ax[1].set_title("Mu")
    ax[2].imshow(tgt, cmap="gray", vmax=vmax_use); ax[2].set_title(title_tgt)
    ax[3].imshow(pred, cmap="gray", vmax=vmax_use);ax[3].set_title(title_pred)
    ax[4].imshow(diff, cmap="seismic");            ax[4].set_title("Diff")
    for a in ax: a.axis("off")

    out_path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[img] saved {out_path}")

def run(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[device] using {device}")

    # outputs
    out_paths = ensure_dirs("/homes/ymellak/PR_Correction/New_Training/Expirements", args.run_name)
    save_args(args, out_paths)

    # data roots
    emission_dir, mumap_dir, stop_dir = resolve_dataset_roots(
        kernel_size=args.kernel_size,
        override=args.data_root
    )
    print(f"[data] emission_dir={emission_dir}")
    print(f"[data] mumap_dir   ={mumap_dir}")
    print(f"[data] stop_dir    ={stop_dir}")

    # dataset kwargs from args
    ds_kwargs = dict(
        emission_ext=args.emission_ext,
        mumap_ext=args.mumap_ext,
        stop_ext=args.stop_ext,
        normalize_prod=bool(args.normalize_prod),
        normalize_stop=bool(args.normalize_stop),
        norm_mode=args.norm_mode,
        eps=args.eps,
        spike_center=bool(args.spike_center),
        spike_value=args.spike_value,
        use_memmap=bool(args.use_memmap),
        return_torch=True,
        torch_dtype=torch.float32,
    )

    # loaders
    train_loader, test_loader = create_dataloaders(
        emission_dir, mumap_dir, stop_dir,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        ds_kwargs=ds_kwargs
    )

    # model / optim / loss
    model = None
    
    if args.model.lower() == "unet":
        model = FilterPredictorUNet(input_channel=2, base=args.base_channels, output_channel=1).to(device) 
    elif args.model.lower() == "fastunet":
        model = FastUNet3D(input_channel=2, base=args.base_channels, output_channel=1).to(device)
    elif args.model.lower() in ["filter", "filterpredictor"]:
        model = FilterPredictor(input_channel=2, in_between_channel=args.base_channels, output_channel=1, num_residual_blocks=4).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion, uses_distribution = make_criterion(args.loss, huber_delta=args.huber_delta)

    # resume if best exists
    start_epoch, best_loss = maybe_load_best(model, optimizer, out_paths["weights"], map_location=device)

    # static distance grid (best-practice: NEGATIVE, NORMALIZED)
    distances = make_distance_grid(
        args.kernel_size, device=device,
        negate=bool(args.negate_dist),
        normalize=args.dist_normalize
    )

    # training
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for it, (images, mu_img, targets) in enumerate(train_loader):
            input_img = images.to(device, non_blocking=True)
            mu_img    = mu_img.to(device, non_blocking=True)
            target_img= targets.to(device, non_blocking=True)

            loss_val = model_train_step(
                model, optimizer, criterion, uses_distribution,
                input_img, mu_img, target_img,
                distances, args.kernel_size
            )
            total_loss += loss_val

            if it % args.log_interval == 0:
                print(f"[train] epoch {epoch} iter {it}/{len(train_loader)} loss={loss_val:.6f}")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[train] epoch {epoch} avg_loss={avg_loss:.6f}")

        # Save checkpoint and maybe best
        save_checkpoint(epoch, model, optimizer, best_loss, out_paths["weights"])
        best_loss = update_best_if_needed(avg_loss, best_loss, epoch, model, optimizer, out_paths["weights"])

        # quick test viz (first batch)
        model.eval()
        with torch.no_grad():
            for images, mu_img, targets in test_loader:
                input_img = images.to(device, non_blocking=True)
                mu_img    = mu_img.to(device, non_blocking=True)
                target_img= targets.to(device, non_blocking=True)
                pred = model_predict(model, input_img, mu_img, distances, args.kernel_size, uses_distribution)
                save_mid_slice_png(out_paths["images"], epoch, input_img, mu_img, target_img, pred, uses_distribution)
                break  # one panel per epoch

        # log scalar
        log_path = os.path.join(out_paths["logs"], "train_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "avg_loss": avg_loss}) + "\n")

    print("[done] training complete")

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train PSF (positron-range) kernel predictor")
    
    p.add_argument("--model", type=str, default="filter",
               choices=["unet", "fastunet", "filter"],
               help="Backbone: 'unet' (previous) or 'fastunet' (proposed fast model)")


    # Core training
    p.add_argument("--kernel_size", type=int, required=True, help="Kernel size N for N x N x N (11, 21, or 31)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--test_batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--base_channels", type=int, default=16, help="UNet base channels")
    p.add_argument("--loss", type=str, default="l1", choices=["l1","l2","huber","kl"],
                   help="Loss to use: l1 (MAE), l2 (MSE), huber (SmoothL1), kl (KL divergence on distributions)")
    p.add_argument("--huber_delta", type=float, default=1.0, help="Delta for Huber (SmoothL1) loss")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=None, help="Limit dataset length for quick runs")
    p.add_argument("--num_workers", type=int, default=4)

    # Data handling
    p.add_argument("--data_root", type=str, default=None,
                   help=("Override dataset root (expects subdirs Emission/, MuMap/, Stop/). "
                         "If not set: kernel_size 11->Dataset_Excrop_2mm, 21->Dataset_crop_2mm, 31->Dataset_2mm."))
    p.add_argument("--emission_ext", type=str, default=".raw")
    p.add_argument("--mumap_ext", type=str, default=".bin")
    p.add_argument("--stop_ext", type=str, default=".raw")

    # DataSetManager2C normalization/spike options
    p.add_argument("--normalize_prod", type=int, default=1, help="1/0 normalize prod (sum or max per --norm_mode)")
    p.add_argument("--normalize_stop", type=int, default=1, help="1/0 normalize stop (sum or max)")
    p.add_argument("--norm_mode", type=str, default="sum", choices=["sum","max"], help="Normalization mode")
    p.add_argument("--eps", type=float, default=1e-12, help="Epsilon to avoid divide-by-zero during normalization")
    p.add_argument("--spike_center", type=int, default=1, help="1/0 spike the center of prod")
    p.add_argument("--spike_value", type=float, default=1e6, help="Value used to spike center of prod (and scale stop if normalized)")
    p.add_argument("--use_memmap", type=int, default=0, help="1/0 use numpy memmap for file loading")

    # Distance-channel options (NEW)
    p.add_argument("--negate_dist", type=int, default=1, help="1/0: use negative distances (recommended=1)")
    p.add_argument("--dist_normalize", type=str, default="max", choices=["max","none"],
                   help="Normalize distance by max radius (recommended='max')")

    # Run & IO
    p.add_argument("--run_name", type=str, default=None, help="Name for output folder under Expirements/")

    args = p.parse_args()

    if args.run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"unet_k{args.kernel_size}_{args.loss}_b{args.base_channels}_{stamp}"

    return args

if __name__ == "__main__":
    args = parse_args()
    run(args)

