#!/usr/bin/env python
"""GT Edge v3: P+D only (N removed from encoder input and edge bias).
N is only used as direction loss target in dir training, not in edge.
Same settings as v2 for fair comparison.

Usage:
  python scripts/train/train_gt_g200_edge_v3.py --seed 0
  python scripts/train/train_gt_g200_edge_v3.py --ensemble
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time, math, argparse
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from boyue.model import GraphTransformerEncoderV3, EdgeHeadV3

# ===========================================================
G = 200
d_model = 512
n_heads = 8
n_layers = 8
dropout = 0.1
sd_prob = 0.1
label_smoothing = 0.1
lr = 2e-4
warmup_steps = 2000
batch_size = 32
total_steps = 20000
log_every = 200
val_every = 2000
grad_clip = 1.0
pos_weight = 10.0

cache_dir = Path(os.environ.get("BOYUE_CACHE", str(PROJECT_ROOT / "processed_data" / "phase1_pdn_cache_g200")))
checkpoint_dir = PROJECT_ROOT / "checkpoints" / "gt_g200_edge_v3"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ===========================================================


class PDNDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob("dataset_*.npz"))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        while True:
            try:
                d = np.load(self.files[idx])
                P = torch.tensor(d['P'], dtype=torch.float32)
                D = torch.tensor(d['D'], dtype=torch.float32)
                N = torch.tensor(d['N'], dtype=torch.float32)  # kept for dir training only
                A = torch.tensor(d['A'], dtype=torch.float32)
                return P, D, N, A
            except Exception:
                idx = (idx + 1) % len(self.files)


@torch.no_grad()
def evaluate(encoder, edge_head, loader):
    encoder.eval()
    edge_head.eval()
    all_probs, all_labels = [], []
    for P, D_mat, N_mat, A in loader:
        P, D_mat = P.to(device), D_mat.to(device)
        A = A.to(device)
        h = encoder(P, D_mat)
        logits = edge_head(h, P, D_mat)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        labels = A.cpu().numpy()
        mask = ~np.eye(G, dtype=bool)
        for b in range(len(probs)):
            all_probs.append(probs[b][mask])
            all_labels.append(labels[b][mask])
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5
    try:
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auprc = 0.0
    return auroc, auprc


def train_one_seed(seed):
    print(f"\n{'#'*60}")
    print(f"# V3 SEED {seed}: P+D encoder, LabelSmooth={label_smoothing} SD={sd_prob}")
    print(f"{'#'*60}")

    full_ds = PDNDataset(cache_dir)
    n_total = len(full_ds)
    n_train, n_val = int(n_total * 0.8), int(n_total * 0.1)
    indices = np.random.RandomState(42 + seed).permutation(n_total)
    train_ds = Subset(full_ds, indices[:n_train])
    val_ds = Subset(full_ds, indices[n_train:n_train + n_val])
    test_ds = Subset(full_ds, indices[n_train + n_val:])

    def collate(batch):
        return (torch.stack([b[0] for b in batch]),
                torch.stack([b[1] for b in batch]),
                torch.stack([b[2] for b in batch]),
                torch.stack([b[3] for b in batch]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate, num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=4, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)

    encoder = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                         n_layers=n_layers, dropout=dropout,
                                         sd_prob=sd_prob).to(device)
    edge_head = EdgeHeadV3(d_model=d_model, d_k=128).to(device)

    n_params = sum(p.numel() for p in list(encoder.parameters()) + list(edge_head.parameters()))
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(edge_head.parameters()),
                                   lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    t_start = time.time()
    step = 0
    loader_iter = iter(train_loader)
    best_val_auroc = 0.0
    seed_dir = checkpoint_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    smooth_pos = 1.0 - label_smoothing
    smooth_neg = label_smoothing

    while step < total_steps:
        try:
            P, D_mat, N_mat, A = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            P, D_mat, N_mat, A = next(loader_iter)

        P, D_mat = P.to(device), D_mat.to(device)
        A = A.to(device)

        optimizer.zero_grad()
        encoder.train()
        edge_head.train()
        with torch.amp.autocast('cuda'):
            h = encoder(P, D_mat)
            logits = edge_head(h, P, D_mat)
            mask = ~torch.eye(G, dtype=torch.bool, device=device).unsqueeze(0).expand_as(A)

            target = A[mask] * smooth_pos + (1 - A[mask]) * smooth_neg
            weight = torch.where(A[mask] > 0.5, pos_weight, 1.0)
            bce = F.binary_cross_entropy_with_logits(
                logits[mask], target, weight=weight, reduction='mean')

        scaler.scale(bce).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(edge_head.parameters()), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for pg in optimizer.param_groups:
            pg['lr'] = get_lr(step)
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t_start
            sps = step / max(elapsed, 1)
            eta = (total_steps - step) / max(sps, 0.001)
            print(f"[seed{seed}][{step:5d}/{total_steps}] loss={bce.item():.4f} "
                  f"lr={get_lr(step):.2e} {sps:.1f}s/s ETA={eta/60:.0f}min")

        if step % val_every == 0:
            val_auroc, val_auprc = evaluate(encoder, edge_head, val_loader)
            best = "*" if val_auroc > best_val_auroc else " "
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                torch.save({'encoder': encoder.state_dict(),
                            'edge_head': edge_head.state_dict()},
                           seed_dir / "best.pt")
            elapsed = time.time() - t_start
            print(f"  >>> SEED{seed} STEP {step}: val_AUROC={val_auroc:.4f} "
                  f"best={best_val_auroc:.4f} {best}  [{elapsed/60:.1f}min]")

    torch.save({'encoder': encoder.state_dict(),
                'edge_head': edge_head.state_dict()}, seed_dir / "final.pt")

    ckpt = torch.load(seed_dir / "best.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder'])
    edge_head.load_state_dict(ckpt['edge_head'])
    test_auroc, test_auprc = evaluate(encoder, edge_head,
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   collate_fn=collate, num_workers=4, pin_memory=True))

    elapsed = time.time() - t_start
    print(f"\nSEED {seed} DONE in {elapsed/60:.1f}min")
    print(f"  val AUROC={best_val_auroc:.4f}  test AUROC={test_auroc:.4f}  AUPRC={test_auprc:.4f}")

    return test_auroc, test_auprc, encoder, edge_head


def ensemble_evaluate(seeds_info, loader):
    encoders, edge_heads = [], []
    for _, _, enc, head in seeds_info:
        enc.eval()
        head.eval()
        encoders.append(enc)
        edge_heads.append(head)

    all_probs, all_labels = [], []
    for P, D_mat, N_mat, A in loader:
        P, D_mat = P.to(device), D_mat.to(device)
        A = A.to(device)
        avg_probs = None
        for enc, head in zip(encoders, edge_heads):
            with torch.no_grad():
                h = enc(P, D_mat)
                logits = head(h, P, D_mat)
                probs = torch.sigmoid(logits.float())
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs += probs
        avg_probs = (avg_probs / len(encoders)).cpu().numpy()
        labels = A.cpu().numpy()
        mask = ~np.eye(G, dtype=bool)
        for b in range(len(avg_probs)):
            all_probs.append(avg_probs[b][mask])
            all_labels.append(labels[b][mask])

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return roc_auc_score(all_labels, all_probs), average_precision_score(all_labels, all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()

    if args.ensemble:
        seed_dirs = sorted(checkpoint_dir.glob("seed_*"))
        if not seed_dirs:
            print("No seeds found.")
            return

        print(f"Loading {len(seed_dirs)} seeds for ensemble...")
        full_ds = PDNDataset(cache_dir)
        n_total = len(full_ds)
        n_train, n_val = int(n_total * 0.8), int(n_total * 0.1)
        indices = np.random.RandomState(42).permutation(n_total)
        test_ds = Subset(full_ds, indices[n_train + n_val:])

        def collate(batch):
            return (torch.stack([b[0] for b in batch]),
                    torch.stack([b[1] for b in batch]),
                    torch.stack([b[2] for b in batch]),
                    torch.stack([b[3] for b in batch]))

        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  collate_fn=collate, num_workers=4, pin_memory=True)

        seeds_info = []
        for sd in seed_dirs:
            seed = int(sd.name.split('_')[1])
            ckpt = torch.load(sd / "best.pt", map_location=device, weights_only=True)
            encoder = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads,
                                                 n_layers=n_layers, dropout=dropout,
                                                 sd_prob=sd_prob).to(device)
            edge_head = EdgeHeadV3(d_model=d_model, d_k=128).to(device)
            encoder.load_state_dict(ckpt['encoder'])
            edge_head.load_state_dict(ckpt['edge_head'])
            seeds_info.append((seed, sd, encoder, edge_head))
            print(f"  Loaded seed {seed}")

        ens_auroc, ens_auprc = ensemble_evaluate(seeds_info, test_loader)
        print(f"\n{'='*60}")
        print(f"ENSEMBLE ({len(seed_dirs)} seeds): AUROC={ens_auroc:.4f}, AUPRC={ens_auprc:.4f}")
        print(f"{'='*60}")

        # Individual scores
        for seed, sd, enc, head in seeds_info:
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate, num_workers=4, pin_memory=True)
            enc.eval(); head.eval()
            auroc, auprc = evaluate(enc, head, test_loader)
            print(f"  seed {seed}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
        return

    if args.seed >= 0:
        train_one_seed(args.seed)
    else:
        print("Usage: --seed N  or  --ensemble")
        print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
