"""PDNDataset: loads P+D+N+A tensors from cache npz files.

Used by training scripts. Each npz contains:
    P: (G, G) precision matrix
    D: (G, G) distance correlation matrix
    N: (G, G) ANM asymmetry (only used as direction loss target)
    A: (G, G) ground-truth adjacency matrix

For inference on real scRNA-seq data, use boyue.stats.compute_PDN to compute
P and D on the fly; N is not needed.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PDNDataset(Dataset):
    """Dataset of cached P+D+N+A tensors for training/validation."""

    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob("dataset_*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Auto-skip corrupted files
        while True:
            try:
                d = np.load(self.files[idx])
                P = torch.tensor(d['P'], dtype=torch.float32)
                D = torch.tensor(d['D'], dtype=torch.float32)
                N = torch.tensor(d['N'], dtype=torch.float32)
                A = torch.tensor(d['A'], dtype=torch.float32)
                return P, D, N, A
            except Exception:
                idx = (idx + 1) % len(self.files)
