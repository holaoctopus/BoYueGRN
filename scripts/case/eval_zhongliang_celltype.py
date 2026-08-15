#!/usr/bin/env python
"""GSE196296 zhongliang: Cell-type-specific GRN inference.

For each viable cell type, run BoYue on each of the 3 PVL samples (A->early, B->mid, C->late).
Compare GRN changes across progression stages.

Model: gt_g200_edge_v3 + gt_g200_dir_specialist_tf_non_tf (seed_0)

Usage:
  python scripts/eval_zhongliang_celltype.py
"""
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import torch
import sys
import time
from pathlib import Path
from sklearn.covariance import LedoitWolf
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Data dir: set BOYUE_DATA env var to point to downloaded datasets
import os
_data_root = Path(os.environ.get("BOYUE_DATA", PROJECT_ROOT / "data_external"))
DATA_DIR = _data_root / "zhongliang"
RESULT_DIR = DATA_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_gt_g200_edge_v3 import (
    GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads,
    n_layers, dropout, sd_prob, device
)

# -- Known human TF database (same as AD) ----------------
KNOWN_TFS = set("""
AHR AR ARNT ARNTL ATF1 ATF2 ATF3 ATF4 ATF5 ATF6 BACH1 BACH2 BATF BATF3
BCL6 BCL11A BCL11B BHLHE40 BHLHE41 BPTF CEBPA CEBPB CEBPD CEBPE CEBPG
CREB1 CREB3 CREB5 CREM CTCF CTCFL DBP DLX1 DLX2 DLX5 DLX6 E2F1 E2F2
E2F3 E2F4 E2F5 E2F6 E2F7 E2F8 EBF1 EBF3 EGR1 EGR2 EGR3 EGR4 ELF1 ELF2 ELF3
ELF4 ELF5 ELK1 ELK3 ELK4 EPAS1 ERF ERG ESR1 ESR2 ESRRA ESRRB ESRRG ETS1
ETS2 ETV1 ETV2 ETV3 ETV4 ETV5 ETV6 ETV7 FEV FOS FOSB FOSL1 FOSL2 FLI1
FOXA1 FOXA2 FOXA3 FOXC1 FOXC2 FOXD1 FOXD3 FOXE1 FOXF1 FOXF2 FOXG1
FOXH1 FOXI1 FOXJ1 FOXJ2 FOXJ3 FOXK1 FOXK2 FOXL1 FOXL2 FOXM1 FOXN1 FOXN3
FOXN4 FOXO1 FOXO3 FOXO4 FOXO6 FOXP1 FOXP2 FOXP3 FOXP4 FOXQ1 FOXS1
GABPA GATA1 GATA2 GATA3 GATA4 GATA5 GATA6 GFI1 GFI1B GLI1 GLI2 GLI3
GRHL1 GRHL2 GRHL3 GTF2I GTF3A HAND1 HAND2 HEY1 HEY2 HEYL HIF1A HIF3A
HINFP HLF HMGA1 HMGA2 HNF1A HNF1B HNF4A HNF4G HOXA1 HOXA2 HOXA3 HOXA4
HOXA5 HOXA6 HOXA7 HOXA9 HOXA10 HOXA11 HOXA13 HOXB1 HOXB2 HOXB3 HOXB4
HOXB5 HOXB6 HOXB7 HOXB8 HOXB9 HOXC4 HOXC5 HOXC6 HOXC8 HOXC9 HOXC10
HOXC11 HOXC12 HOXC13 HOXD1 HOXD3 HOXD4 HOXD8 HOXD9 HOXD10 HOXD11 HOXD12
HOXD13 HSF1 HSF2 HSF4 IRF1 IRF2 IRF3 IRF4 IRF5 IRF6 IRF7 IRF8 IRF9
IRX3 IRX5 JUN JUNB JUND KLF1 KLF2 KLF3 KLF4 KLF5 KLF6 KLF7 KLF8 KLF9
KLF10 KLF11 KLF12 KLF13 KLF14 KLF15 KLF16 KLF17 LEF1 LHX2 LHX3 LHX4 LHX5
LHX6 LHX8 MAF MAFA MAFB MAFF MAFG MAFK MAX MAZ MECP2 MECOM MED1 MEF2A
MEF2B MEF2C MEF2D MEIS1 MEIS2 MEIS3 MEOX1 MEOX2 MITF MIXL1 MKX MLX
MLXIP MLXIPL MSC MSX1 MSX2 MXD1 MXD3 MXD4 MYB MYBL1 MYBL2 MYC MYCN
MYF5 MYF6 MYOD1 MYOG MZF1 NANOG NEUROD1 NEUROD2 NEUROD4 NEUROD6 NEUROG1
NEUROG2 NFAT5 NFATC1 NFATC2 NFATC3 NFATC4 NFE2 NFE2L1 NFE2L2 NFE2L3
NFIA NFIB NFIC NFIL3 NFIX NFKB1 NFKB2 NFYA NFYB NFYC NKX2-1 NKX2-2
NKX2-3 NKX2-5 NKX3-1 NKX3-2 NKX6-1 NKX6-2 NKX6-3 NOTCH1 NOTCH2 NOTCH3
NOTCH4 NR0B1 NR0B2 NR1D1 NR1D2 NR1H2 NR1H3 NR1H4 NR1I2 NR1I3 NR2C1
NR2C2 NR2E1 NR2E3 NR2F1 NR2F2 NR2F6 NR3C1 NR3C2 NR4A1 NR4A2 NR4A3
NR5A1 NR5A2 NR6A1 OLIG1 OLIG2 OSR1 OSR2 OVOL1 OVOL2 OVOL3 PAX2 PAX3 PAX4
PAX5 PAX6 PAX7 PAX8 PAX9 PBX1 PBX2 PBX3 PBX4 PDX1 PEG3 PHOX2A PHOX2B
PITX1 PITX2 PITX3 PKNOX1 PKNOX2 PLAG1 PLAGL1 PLAGL2 PLSCR1 PLSCR4 POU1F1
POU2F1 POU2F2 POU3F1 POU3F2 POU3F3 POU3F4 POU4F1 POU4F2 POU4F3 POU5F1
POU6F1 POU6F2 PPARA PPARD PPARG PPARGC1A PPARGC1B PRDM1 PRDM2 PRDM4
PRDM5 PRDM6 PRDM14 PROX1 PRRX1 PRRX2 RARA RARB RARG RB1 RBL1 RBL2 RELA
RELB RERE REST RFX1 RFX2 RFX3 RFX4 RFX5 RFX7 RFX8 RORA RORB RORC RUNX1
RUNX2 RUNX3 RXRA RXRB RXRG SALL1 SALL2 SALL3 SALL4 SATB1 SATB2 SCRT1
SCRT2 SHOX SHOX2 SIM1 SIM2 SIX1 SIX2 SIX3 SIX4 SIX5 SKI SKIL SMAD1 SMAD2
SMAD3 SMAD4 SMAD5 SMAD6 SMAD7 SNAI1 SNAI2 SNAI3 SOX1 SOX2 SOX3 SOX4 SOX5
SOX6 SOX7 SOX8 SOX9 SOX10 SOX11 SOX12 SOX13 SOX14 SOX15 SOX17 SOX18
SOX21 SOX30 SP1 SP2 SP3 SP4 SP5 SP6 SP7 SP8 SP100 SP110 SP140 SPI1 SPIB
SREBF1 SREBF2 SRF SRY STAT1 STAT2 STAT3 STAT4 STAT5A STAT5B STAT6 T
TFAP2A TFAP2B TFAP2C TFAP2D TFAP2E TFAP4 TBP TBPL1 TBPL2 TBX1 TBX2 TBX3
TBX4 TBX5 TBX6 TBX10 TBX15 TBX18 TBX19 TBX20 TBX21 TBX22 TCF3 TCF4 TCF7
TCF7L1 TCF7L2 TCF12 TCF15 TCF21 TCF23 TCF25 TEAD1 TEAD2 TEAD3 TEAD4
TFDP1 TFDP2 TFDP3 TFEB THRA THRB TLX1 TLX2 TLX3 TP53 TP63 TP73 TSHZ1
TSHZ2 TSHZ3 USF1 USF2 VAX1 VAX2 VDR VEZF1 WT1 YBX1 YY1 YY2 ZBTB1 ZBTB2
ZBTB3 ZBTB5 ZBTB7A ZBTB7B ZBTB7C ZBTB10 ZBTB11 ZBTB14 ZBTB16 ZBTB17
ZBTB18 ZBTB20 ZBTB21 ZBTB22 ZBTB24 ZBTB25 ZBTB26 ZBTB32 ZBTB33 ZBTB34
ZBTB37 ZBTB38 ZBTB39 ZBTB40 ZBTB41 ZBTB42 ZBTB43 ZBTB44 ZBTB45 ZBTB46
ZBTB47 ZBTB48 ZBTB49 ZEB1 ZEB2 ZFP36 ZFP36L1 ZFP36L2 ZFPM1 ZFPM2 ZFX
ZFY ZIC1 ZIC2 ZIC3 ZIC4 ZIC5 ZKSCAN1 ZKSCAN3
""".split())
KNOWN_TFS = {t for t in KNOWN_TFS if t and len(t) > 1}

N_TFS = 50; N_TARGETS = 150; THRESHOLD = 0.2; MAX_CELLS = 1000
SAMPLES = ['A', 'B', 'C']

# -- Model loading ----------------------------------
print("Loading BoYue models...")
edge_ckpt = PROJECT_ROOT / "checkpoints" / "main" / "edge_v3_seed0.pt"
dir_ckpt = PROJECT_ROOT / "checkpoints" / "main" / "dir_specialist_tf_non_tf_seed0.pt"

encoder = GraphTransformerEncoderV3(G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers, sd_prob=0.0).to(device)
edge_head = EdgeHeadV3(d_model=d_model).to(device)
state = torch.load(edge_ckpt, map_location=device, weights_only=True)
encoder.load_state_dict(state['encoder']); edge_head.load_state_dict(state['edge_head'])
encoder.eval(); edge_head.eval()

class AsymDirHead(torch.nn.Module):
    def __init__(self, dm=512):
        super().__init__()
        self.src_proj = torch.nn.Sequential(torch.nn.Linear(dm, dm), torch.nn.GELU(), torch.nn.Linear(dm, dm))
        self.tgt_proj = torch.nn.Sequential(torch.nn.Linear(dm, dm), torch.nn.GELU(), torch.nn.Linear(dm, dm))
        self.edge_net = torch.nn.Sequential(torch.nn.Linear(2, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))
    def forward(self, h, P, D):
        s = torch.bmm(self.src_proj(h), self.tgt_proj(h).transpose(1, 2))
        ef = torch.cat([P.unsqueeze(-1), D.unsqueeze(-1)], dim=-1)
        return s + self.edge_net(ef).squeeze(-1)

dir_head = AsymDirHead().to(device)
dstate = torch.load(dir_ckpt, map_location=device, weights_only=True)
dir_head.load_state_dict(dstate['dir_head']); dir_head.eval()
print("  Ready.\n")

# -- Data -------------------------------------------
print("Loading annotated data...")
adata = sc.read_h5ad(DATA_DIR / "processed" / "zhongliang_celltype.h5ad")
print(f"  {adata.shape[0]} cells, {adata.shape[1]} genes\n")

# -- Functions --------------------------------------
def compute_P_D(X):
    n_cells, Gv = X.shape
    lw = LedoitWolf().fit(X)
    P = lw.precision_
    np.fill_diagonal(P, 0)
    
    D_matrices = np.zeros((Gv, n_cells, n_cells), dtype=np.float32)
    for g in range(Gv):
        x = X[:, g].reshape(-1, 1)
        dists = np.abs(x - x.T)
        row_mean = dists.mean(axis=1, keepdims=True)
        col_mean = dists.mean(axis=0, keepdims=True)
        grand_mean = dists.mean()
        D_matrices[g] = dists - row_mean - col_mean + grand_mean
    
    D_flat = D_matrices.reshape(Gv, -1)
    dCov_sq = (D_flat @ D_flat.T) / (n_cells * n_cells)
    dCov_sq = np.clip(dCov_sq, 0, None)
    dVar = np.diag(dCov_sq).copy()
    dVar[dVar < 1e-12] = 1e-12
    D = dCov_sq / np.sqrt(np.outer(dVar, dVar))
    np.fill_diagonal(D, 0)
    D = np.clip(D, -1, 1)
    return P, D

@torch.no_grad()
def infer_grn(P, D):
    P_t = torch.tensor(P, dtype=torch.float32).unsqueeze(0).to(device)
    D_t = torch.tensor(D, dtype=torch.float32).unsqueeze(0).to(device)
    h = encoder(P_t, D_t)
    edge_logits = edge_head(h, P_t, D_t)
    dir_logits = dir_head(h, P_t, D_t)
    edge_prob = torch.sigmoid(edge_logits).squeeze(0).cpu().numpy()
    direction = (torch.sigmoid(dir_logits) > 0.5).float().squeeze(0).cpu().numpy()
    dir_score = torch.sigmoid(dir_logits).squeeze(0).cpu().numpy()
    np.fill_diagonal(edge_prob, 0); np.fill_diagonal(direction, 0)
    return edge_prob, dir_score, direction

# -- Select viable cell types ------------------------
cell_type_counts = adata.obs['cell_type'].value_counts()
viable_types = []
print("Cell type viability:")
for ct in sorted(cell_type_counts.index):
    counts = {s: ((adata.obs['cell_type']==ct) & (adata.obs['sample']==s)).sum() for s in SAMPLES}
    viable = all(c >= 100 for c in counts.values())
    print(f"  {ct:<12s}: A={counts['A']:>5d}  B={counts['B']:>5d}  C={counts['C']:>5d}  {'[OK]' if viable else '[X] <100 in some samples'}")
    if viable:
        viable_types.append(ct)

print(f"\nViable: {viable_types}")

# -- Main loop --------------------------------------
all_results = {}
t_start = time.time()

for ct in viable_types:
    print(f"\n{'='*55}")
    print(f"  {ct}")
    print(f"{'='*55}")
    
    ct_data = adata[adata.obs['cell_type'] == ct].copy()
    expr_ct = ct_data.X.toarray() if hasattr(ct_data.X, 'toarray') else ct_data.X
    
    # Select genes based on combined variance
    gene_vars = np.var(expr_ct, axis=0)
    all_genes = list(ct_data.var_names)
    known_tf_in_data = [(i, g) for i, g in enumerate(all_genes) if g in KNOWN_TFS]
    
    # Select top TFs by variance
    tf_vars = [(gene_vars[i], g, i) for i, g in known_tf_in_data]
    tf_vars.sort(key=lambda x: -x[0])
    selected_tfs = tf_vars[:N_TFS]
    tf_genes = [t[1] for t in selected_tfs]
    tf_indices = set(t[2] for t in selected_tfs)
    
    # Select targets from non-TF genes
    non_tf = [(gene_vars[i], g, i) for i, g in enumerate(all_genes) 
              if i not in tf_indices and g not in KNOWN_TFS]
    non_tf.sort(key=lambda x: -x[0])
    selected_targets = non_tf[:N_TARGETS]
    
    selected_indices = [t[2] for t in selected_tfs] + [t[2] for t in selected_targets]
    selected_gene_names = [t[1] for t in selected_tfs] + [t[1] for t in selected_targets]
    
    print(f"  {N_TFS} TFs + {N_TARGETS} targets. Top TFs: {tf_genes[:6]}")
    
    expr_selected = expr_ct[:, selected_indices]
    ct_results = {}
    
    for sample in SAMPLES:
        mask = ct_data.obs['sample'] == sample
        X = expr_selected[mask]
        n_cells = X.shape[0]
        
        print(f"  Sample {sample}: {n_cells:,} cells", end='')
        t0 = time.time()
        
        if n_cells > MAX_CELLS:
            rng = np.random.RandomState(42)
            X = X[rng.choice(n_cells, MAX_CELLS, replace=False)]
            print(f" (subsampled {MAX_CELLS})", end='')
        
        P, D = compute_P_D(X)
        edge_prob, dir_score, direction = infer_grn(P, D)
        
        n_edges = int((edge_prob > THRESHOLD).sum())
        elapsed = time.time() - t0
        print(f" -> {n_edges:,} edges ({elapsed:.1f}s)")
        
        ct_results[sample] = {
            'genes': selected_gene_names,
            'tf_genes': tf_genes,
            'edge_prob': edge_prob,
            'dir_score': dir_score,
            'direction': direction,
            'n_cells': n_cells,
        }
    
    all_results[ct] = {'tf_genes': tf_genes, 'results': ct_results}

# -- Save -------------------------------------------
out_path = RESULT_DIR / "zhongliang_celltype.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(all_results, f)

total_elapsed = time.time() - t_start
print(f"\n{'='*55}")
print(f"ALL DONE in {total_elapsed/60:.1f} min")
print(f"Saved to {out_path}")

# -- Summary -----------------------------------------
print(f"\n{'Cell Type':<15s} {'A':>10s} {'B':>10s} {'C':>10s} {'B-A':>10s} {'C-B':>10s}")
print("-" * 60)
for ct in viable_types:
    if ct not in all_results: continue
    r = all_results[ct]['results']
    ae = int((r['A']['edge_prob'] > THRESHOLD).sum())
    be = int((r['B']['edge_prob'] > THRESHOLD).sum())
    ce = int((r['C']['edge_prob'] > THRESHOLD).sum())
    print(f"  {ct:<15s} {ae:>10,d} {be:>10,d} {ce:>10,d} {be-ae:>+10,d} {ce-be:>+10,d}")
