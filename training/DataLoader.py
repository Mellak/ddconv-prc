import os
import numpy as np
from torch.utils.data import Dataset
import torch
import re

class DataSetManager(Dataset):
    def __init__(self, emission_dir, mumap_dir, stop_dir, max_samples=2000, img_size=(61, 61, 61)):
        self.emission_dir = emission_dir
        self.mumap_dir = mumap_dir
        self.stop_dir = stop_dir
        self.img_size = img_size
        self.max_samples = max_samples
        
        # List all Emission files (assuming the files are named as Emission_{idx}.raw)
        self.file_list = sorted([file for file in os.listdir(emission_dir) if file.startswith('Emission_') and file.endswith('.raw')], 
                                key=lambda x: int(x.split('_')[1].split('.')[0]))[:max_samples]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # Extract the index from the filename
        file_name = self.file_list[index]
        image_number = file_name.split('_')[1].split('.')[0]  # Extract the {idx} part

        # Construct full paths for Emission, MuMap, and Stop files
        emission_file = os.path.join(self.emission_dir, f"Emission_{image_number}.raw")
        mumap_file = os.path.join(self.mumap_dir, f"MuMap_{image_number}.bin")
        stop_file = os.path.join(self.stop_dir, f"Stop_{image_number}.raw")

        # Load the files into numpy arrays
        prod_image = np.fromfile(emission_file, dtype=np.float32).reshape(self.img_size)
        mu_image = np.fromfile(mumap_file, dtype=np.float32).reshape(self.img_size)
        stop_image = np.fromfile(stop_file, dtype=np.float32).reshape(self.img_size)

        # unsqueeze the image to add a channel dimension
        prod_image = np.expand_dims(prod_image, axis=0)
        mu_image = np.expand_dims(mu_image, axis=0)
        stop_image = np.expand_dims(stop_image, axis=0)

        # make the lowest value 0, the second lowest 1, and so on
        unique_values = np.unique(mu_image)
        for i, value in enumerate(unique_values):
            mu_image[mu_image == value] = i+1
        

        # Return the emission (Prod), mu_map (MuMap), and stop data
        return prod_image, mu_image, stop_image
    

class DataSetManager2C(Dataset):
    """
    Loads triples (prod_image, mu_image, stop_image) with shape [1, D, H, W] each.
    - File names assumed: Emission_<idx>.<ext>, MuMap_<idx>.<ext>, Stop_<idx>.<ext>
    - Normalization and center spike are configurable.
    """
    def __init__(
        self,
        emission_dir: str,
        mumap_dir: str,
        stop_dir: str,
        max_samples: int = None,
        img_size=(11, 11, 11),
        emission_ext: str = ".raw",
        mumap_ext: str = ".bin",
        stop_ext: str = ".raw",
        normalize_prod: bool = True,
        normalize_stop: bool = True,
        norm_mode: str = "sum",       # "sum" or "max"
        eps: float = 1e-12,
        spike_center: bool = True,
        spike_value: float = 1,
        use_memmap: bool = False,     # optional; enable if files are huge
        return_torch: bool = True,
        torch_dtype=torch.float32,
    ):
        self.emission_dir = emission_dir
        self.mumap_dir = mumap_dir
        self.stop_dir = stop_dir
        self.img_size = tuple(img_size)
        self.emission_ext = emission_ext
        self.mumap_ext = mumap_ext
        self.stop_ext = stop_ext
        self.normalize_prod = normalize_prod
        self.normalize_stop = normalize_stop
        self.norm_mode = norm_mode
        self.eps = eps
        self.spike_center = spike_center
        self.spike_value = spike_value
        self.use_memmap = use_memmap
        self.return_torch = return_torch
        self.torch_dtype = torch_dtype

        D, H, W = self.img_size
        assert D > 0 and H > 0 and W > 0, "img_size must be positive ints"

        # Validate dirs
        for d in (emission_dir, mumap_dir, stop_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory not found: {d}")

        # Build file list from emission_dir; require pattern Emission_<idx>.<ext>
        pat = re.compile(rf"^Emission_(\d+){re.escape(emission_ext)}$")
        all_files = []
        for f in os.listdir(emission_dir):
            m = pat.match(f)
            if m:
                idx = int(m.group(1))
                all_files.append((idx, f))
        if not all_files:
            raise FileNotFoundError(f"No files matching Emission_<idx>{emission_ext} in {emission_dir}")

        # Sort by idx and cap to max_samples
        all_files.sort(key=lambda x: x[0])
        if max_samples is not None:
            all_files = all_files[:max_samples]

        self.indices = [idx for idx, _ in all_files]
        self.file_list = [fname for _, fname in all_files]

        # Precompute center index
        self.cz, self.cy, self.cx = D // 2, H // 2, W // 2
        self._nvox = D * H * W
        self._dtype = np.float32

    def __len__(self):
        return len(self.file_list)

    def _safe_load(self, path: str, shape, use_memmap: bool):
        """Load a binary float32 array and reshape; can use memmap for large files."""
        D, H, W = shape
        n = D * H * W
        if use_memmap:
            arr = np.memmap(path, dtype=self._dtype, mode="r", shape=(n,))
        else:
            arr = np.fromfile(path, dtype=self._dtype, count=n)
        if arr.size != n:
            raise ValueError(f"File {path} has {arr.size} floats, expected {n} for shape {shape}")
        arr = np.asarray(arr, dtype=self._dtype).reshape(shape, order="C")
        return arr

    def _normalize(self, arr: np.ndarray, mode: str):
        if mode == "sum":
            s = float(arr.sum())
            if s > self.eps:
                return arr / s
            else:
                # avoid NaNs; if zero-sum, just return zeros
                return np.zeros_like(arr)
        elif mode == "max":
            m = float(np.max(np.abs(arr)))
            if m > self.eps:
                return arr / m
            else:
                return np.zeros_like(arr)
        else:
            raise ValueError(f"Unsupported norm_mode: {mode}")

    def __getitem__(self, index):
        # resolve idx from emission filename
        emission_name = self.file_list[index]
        idx_str = emission_name.split("_")[1].split(".")[0]

        emit_path = os.path.join(self.emission_dir, f"Emission_{idx_str}{self.emission_ext}")
        mu_path   = os.path.join(self.mumap_dir,    f"MuMap_{idx_str}{self.mumap_ext}")
        stop_path = os.path.join(self.stop_dir,     f"Stop_{idx_str}{self.stop_ext}")

        # I/O
        prod = self._safe_load(emit_path, self.img_size, self.use_memmap)
        mu   = self._safe_load(mu_path,   self.img_size, self.use_memmap)
        stop = self._safe_load(stop_path, self.img_size, self.use_memmap)

        # (Optional) normalize prod/stop BEFORE spiking
        if self.normalize_prod:
            prod = self._normalize(prod, self.norm_mode)
        if self.normalize_stop:
            stop = self._normalize(stop, self.norm_mode)

        # Spike the exact center of prod (after normalization), if requested
        if self.spike_center:
            prod[self.cz, self.cy, self.cx] = self.spike_value

        # Scale stop to same magnitude (optional – you currently do this)
        if self.spike_center and self.spike_value is not None and self.normalize_stop:
            stop = stop * self.spike_value

        # Add channel dim
        prod = np.expand_dims(prod, axis=0)   # [1,D,H,W]
        mu   = np.expand_dims(mu,   axis=0)
        stop = np.expand_dims(stop, axis=0)

        if self.return_torch:
            prod = torch.as_tensor(prod, dtype=self.torch_dtype)
            mu   = torch.as_tensor(mu,   dtype=self.torch_dtype)
            stop = torch.as_tensor(stop, dtype=self.torch_dtype)

        return prod, mu, stop
    
if __name__ == '__main__':
    # Define the directories
    emission_dir = "/home/youness/data/PR_Correction/DataSet/Dataset_2mm/Emission/"
    mumap_dir = "/home/youness/data/PR_Correction/DataSet/Dataset_2mm/MuMap/"
    stop_dir = "/home/youness/data/PR_Correction/DataSet/Dataset_2mm/Stop/"

    # Use DataSetManager2C to load data
    dataset = DataSetManager2C(emission_dir, mumap_dir, stop_dir, max_samples=10, img_size=(31, 31, 31))
    for i in range(len(dataset)):
        prod, mu, stop = dataset[i]
        print(f"Sample {i}: Prod shape: {prod.shape}, Mu shape: {mu.shape}, Stop shape: {stop.shape}")
        # print sum of prod and stop to verify normalization
        print(f"  Prod sum: {prod.sum().item()}, Stop sum: {stop.sum().item()}")



