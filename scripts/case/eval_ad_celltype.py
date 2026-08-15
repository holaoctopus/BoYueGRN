#!/usr/bin/env python
"""GSE157827 AD: Cell-type-specific GRN inference (all 8 cell types).

For each cell type (ExcNeuron, InhNeuron, Astrocyte, Oligo, OPC, Microglia, 
Endothelial, Pericyte):
  1. Select top 50 TFs + 150 targets using known TF database + variance
  2. Compute P/D matrices for AD and Control separately
  3. Run BoYue edge + direction inference
  4. Save results

Model: gt_g200_edge_v3 + gt_g200_dir_specialist_tf_non_tf (seed_0)

Usage:
  python scripts/eval_ad_celltype.py
"""
import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import torch
import sys
import os
import time
from pathlib import Path
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Data dir: set BOYUE_DATA env var to point to downloaded datasets
import os
_data_root = Path(os.environ.get("BOYUE_DATA", PROJECT_ROOT / "data_external"))
DATA_DIR = _data_root / "GSE157827"
RESULT_DIR = DATA_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_gt_g200_edge_v3 import (
    GraphTransformerEncoderV3, EdgeHeadV3, G, d_model, n_heads,
    n_layers, dropout, sd_prob, device
)

# -- Known human TF database (DoRothEA + RegNetwork consensus) --
KNOWN_TFS = set("""
AHR AR ARNT ARNTL ATF1 ATF2 ATF3 ATF4 ATF5 ATF6 BACH1 BACH2 BATF BATF3
BCL6 BCL11A BCL11B BHLHE40 BHLHE41 BPTF CEBPA CEBPB CEBPD CEBPE CEBPG
CREB1 CREB3 CREB5 CREM CTCF CTCFL DBF4 DBP DLX1 DLX2 DLX5 DLX6 E2F1 E2F2
E2F3 E2F4 E2F5 E2F6 E2F7 E2F8 EBF1 EBF3 EGR1 EGR2 EGR3 EGR4 ELF1 ELF2 ELF3
ELF4 ELF5 ELK1 ELK3 ELK4 EPAS1 ERF ERG ESR1 ESR2 ESRRA ESRRB ESRRG ETS1
ETS2 ETV1 ETV2 ETV3 ETV4 ETV5 ETV6 ETV7 FEV FOS FOSB FOSL1 FOSL2 FLI1
FOXA1 FOXA2 FOXA3 FOXC1 FOXC2 FOXD1 FOXD2 FOXD3 FOXE1 FOXF1 FOXF2 FOXG1
FOXH1 FOXI1 FOXJ1 FOXJ2 FOXJ3 FOXK1 FOXK2 FOXL1 FOXL2 FOXM1 FOXN1 FOXN2
FOXN3 FOXN4 FOXO1 FOXO3 FOXO4 FOXO6 FOXP1 FOXP2 FOXP3 FOXP4 FOXQ1 FOXS1
GABPA GATA1 GATA2 GATA3 GATA4 GATA5 GATA6 GFI1 GFI1B GLI1 GLI2 GLI3 GLIS1
GLIS2 GRHL1 GRHL2 GRHL3 GTF2I GTF3A HAND1 HAND2 HEY1 HEY2 HEYL HIF1A HIF3A
HINFP HLF HMGA1 HMGA2 HNF1A HNF1B HNF4A HNF4G HOXA1 HOXA2 HOXA3 HOXA4
HOXA5 HOXA6 HOXA7 HOXA9 HOXA10 HOXA11 HOXA13 HOXB1 HOXB2 HOXB3 HOXB4
HOXB5 HOXB6 HOXB7 HOXB8 HOXB9 HOXC4 HOXC5 HOXC6 HOXC8 HOXC9 HOXC10
HOXC11 HOXC12 HOXC13 HOXD1 HOXD3 HOXD4 HOXD8 HOXD9 HOXD10 HOXD11 HOXD12
HOXD13 HSF1 HSF2 HSF4 HSFY1 IRF1 IRF2 IRF3 IRF4 IRF5 IRF6 IRF7 IRF8 IRF9
IRX3 IRX5 JUN JUNB JUND KLF1 KLF2 KLF3 KLF4 KLF5 KLF6 KLF7 KLF8 KLF9
KLF10 KLF11 KLF12 KLF13 KLF14 KLF15 KLF16 KLF17 LEF1 LHX2 LHX3 LHX4 LHX5
LHX6 LHX8 LMX1A LMX1B MAF MAFA MAFB MAFF MAFG MAFK MAX MAZ MECP2 MECOM
MED1 MEF2A MEF2B MEF2C MEF2D MEIS1 MEIS2 MEIS3 MEOX1 MEOX2 MITF MIXL1
MKX MLX MLXIP MLXIPL MSC MSX1 MSX2 MXD1 MXD3 MXD4 MYB MYBL1 MYBL2 MYC
MYCN MYF5 MYF6 MYOD1 MYOG MZF1 NANOG NEUROD1 NEUROD2 NEUROD4 NEUROD6
NEUROG1 NEUROG2 NFAT5 NFATC1 NFATC2 NFATC3 NFATC4 NFE2 NFE2L1 NFE2L2
NFE2L3 NFIA NFIB NFIC NFIL3 NFIX NFKB1 NFKB2 NFYA NFYB NFYC NKX2-1 NKX2-2
NKX2-3 NKX2-5 NKX2-6 NKX3-1 NKX3-2 NKX6-1 NKX6-2 NKX6-3 NOTCH1 NOTCH2
NOTCH3 NOTCH4 NR0B1 NR0B2 NR1D1 NR1D2 NR1H2 NR1H3 NR1H4 NR1I2 NR1I3
NR2C1 NR2C2 NR2E1 NR2E3 NR2F1 NR2F2 NR2F6 NR3C1 NR3C2 NR4A1 NR4A2 NR4A3
NR5A1 NR5A2 NR6A1 OLIG1 OLIG2 OSR1 OSR2 OVOL1 OVOL2 OVOL3 PAX2 PAX3 PAX4
PAX5 PAX6 PAX7 PAX8 PAX9 PBX1 PBX2 PBX3 PBX4 PDX1 PEG3 PHOX2A PHOX2B
PITX1 PITX2 PITX3 PKNOX1 PKNOX2 PLAG1 PLAGL1 PLAGL2 PLSCR1 PLSCR4 POU1F1
POU2F1 POU2F2 POU3F1 POU3F2 POU3F3 POU3F4 POU4F1 POU4F2 POU4F3 POU5F1
POU5F1B POU6F1 POU6F2 PPARA PPARD PPARG PPARGC1A PPARGC1B PRDM1 PRDM2
PRDM4 PRDM5 PRDM6 PRDM14 PROX1 PRRX1 PRRX2 RARA RARB RARG RB1 RBL1 RBL2
RELA RELB RERE REST RFX1 RFX2 RFX3 RFX4 RFX5 RFX7 RFX8 RORA RORB RORC
RUNX1 RUNX2 RUNX3 RXRA RXRB RXRG SALL1 SALL2 SALL3 SALL4 SATB1 SATB2
SCRT1 SCRT2 SHOX SHOX2 SIM1 SIM2 SIX1 SIX2 SIX3 SIX4 SIX5 SKI SKIL SMAD1
SMAD2 SMAD3 SMAD4 SMAD5 SMAD6 SMAD7 SNAI1 SNAI2 SNAI3 SOX1 SOX2 SOX3 SOX4
SOX5 SOX6 SOX7 SOX8 SOX9 SOX10 SOX11 SOX12 SOX13 SOX14 SOX15 SOX17 SOX18
SOX21 SOX30 SP1 SP2 SP3 SP4 SP5 SP6 SP7 SP8 SP100 SP110 SP140 SPI1 SPIB
SREBF1 SREBF2 SRF SRY STAT1 STAT2 STAT3 STAT4 STAT5A STAT5B STAT6 T TFAP2A
TFAP2B TFAP2C TFAP2D TFAP2E TFAP4 TBP TBPL1 TBPL2 TBX1 TBX2 TBX3 TBX4
TBX5 TBX6 TBX10 TBX15 TBX18 TBX19 TBX20 TBX21 TBX22 TCF3 TCF4 TCF7 TCF7L1
TCF7L2 TCF12 TCF15 TCF21 TCF23 TCF25 TEAD1 TEAD2 TEAD3 TEAD4 TFDP1 TFDP2
TFDP3 TFEB THRA THRB TLX1 TLX2 TLX3 TP53 TP63 TP73 TSHZ1 TSHZ2 TSHZ3 USF1
USF2 VAX1 VAX2 VDR VEZF1 VSX1 VSX2 WT1 YBX1 YY1 YY2 ZBTB1 ZBTB2 ZBTB3
ZBTB5 ZBTB7A ZBTB7B ZBTB7C ZBTB10 ZBTB11 ZBTB12 ZBTB14 ZBTB16 ZBTB17
ZBTB18 ZBTB20 ZBTB21 ZBTB22 ZBTB24 ZBTB25 ZBTB26 ZBTB32 ZBTB33 ZBTB34
ZBTB37 ZBTB38 ZBTB39 ZBTB40 ZBTB41 ZBTB42 ZBTB43 ZBTB44 ZBTB45 ZBTB46
ZBTB47 ZBTB48 ZBTB49 ZBTB80 ZEB1 ZEB2 ZFP1 ZFP2 ZFP3 ZFP28 ZFP30 ZFP36
ZFP36L1 ZFP36L2 ZFP37 ZFP41 ZFP42 ZFP57 ZFP62 ZFP64 ZFP69 ZFP82 ZFP90
ZFP91 ZFPM1 ZFPM2 ZFX ZFY ZGLP1 ZIC1 ZIC2 ZIC3 ZIC4 ZIC5 ZKSCAN1 ZKSCAN3
ZNF2 ZNF3 ZNF8 ZNF10 ZNF12 ZNF16 ZNF18 ZNF19 ZNF22 ZNF23 ZNF24 ZNF25
ZNF26 ZNF28 ZNF30 ZNF33A ZNF33B ZNF34 ZNF35 ZNF37A ZNF41 ZNF43 ZNF44
ZNF45 ZNF48 ZNF57 ZNF66 ZNF69 ZNF71 ZNF74 ZNF75A ZNF75D ZNF76 ZNF77 ZNF79
ZNF80 ZNF81 ZNF83 ZNF84 ZNF85 ZNF90 ZNF91 ZNF92 ZNF93 ZNF98 ZNF99 ZNF100
ZNF101 ZNF107 ZNF114 ZNF117 ZNF121 ZNF124 ZNF131 ZNF132 ZNF133 ZNF134
ZNF135 ZNF136 ZNF137 ZNF138 ZNF140 ZNF141 ZNF142 ZNF143 ZNF146 ZNF148
ZNF154 ZNF155 ZNF157 ZNF160 ZNF165 ZNF169 ZNF174 ZNF175 ZNF177 ZNF180
ZNF181 ZNF182 ZNF184 ZNF185 ZNF189 ZNF192 ZNF195 ZNF197 ZNF200 ZNF202
ZNF204 ZNF205 ZNF207 ZNF208 ZNF211 ZNF212 ZNF213 ZNF214 ZNF215 ZNF217
ZNF219 ZNF221 ZNF222 ZNF223 ZNF224 ZNF225 ZNF226 ZNF227 ZNF228 ZNF229
ZNF230 ZNF232 ZNF233 ZNF234 ZNF235 ZNF236 ZNF239 ZNF248 ZNF250 ZNF251
ZNF253 ZNF254 ZNF256 ZNF257 ZNF260 ZNF263 ZNF264 ZNF266 ZNF267 ZNF268
ZNF271 ZNF273 ZNF274 ZNF275 ZNF276 ZNF277 ZNF280A ZNF280B ZNF280C ZNF280D
ZNF281 ZNF282 ZNF283 ZNF284 ZNF285 ZNF286A ZNF287 ZNF292 ZNF296 ZNF300
ZNF302 ZNF304 ZNF311 ZNF316 ZNF317 ZNF318 ZNF319 ZNF320 ZNF322 ZNF324
ZNF326 ZNF329 ZNF330 ZNF331 ZNF333 ZNF334 ZNF335 ZNF337 ZNF338 ZNF341
ZNF343 ZNF345 ZNF346 ZNF347 ZNF350 ZNF354A ZNF354B ZNF354C ZNF355P ZNF358
ZNF362 ZNF365 ZNF366 ZNF367 ZNF382 ZNF383 ZNF384 ZNF385A ZNF385B ZNF385C
ZNF385D ZNF386 ZNF391 ZNF394 ZNF395 ZNF396 ZNF397 ZNF398 ZNF404 ZNF407
ZNF408 ZNF410 ZNF414 ZNF415 ZNF416 ZNF417 ZNF418 ZNF419 ZNF420 ZNF423
ZNF425 ZNF426 ZNF428 ZNF429 ZNF430 ZNF431 ZNF432 ZNF433 ZNF436 ZNF438
ZNF439 ZNF440 ZNF441 ZNF442 ZNF443 ZNF444 ZNF445 ZNF446 ZNF449 ZNF451
ZNF454 ZNF460 ZNF461 ZNF462 ZNF467 ZNF468 ZNF469 ZNF470 ZNF471 ZNF473
ZNF474 ZNF479 ZNF480 ZNF483 ZNF484 ZNF485 ZNF486 ZNF487 ZNF488 ZNF490
ZNF491 ZNF492 ZNF493 ZNF494 ZNF496 ZNF497 ZNF498 ZNF500 ZNF501 ZNF502
ZNF503 ZNF504 ZNF506 ZNF507 ZNF510 ZNF511 ZNF512 ZNF513 ZNF514 ZNF516
ZNF517 ZNF518A ZNF518B ZNF519 ZNF521 ZNF524 ZNF525 ZNF526 ZNF527 ZNF528
ZNF529 ZNF530 ZNF532 ZNF534 ZNF536 ZNF540 ZNF541 ZNF543 ZNF544 ZNF546
ZNF547 ZNF548 ZNF549 ZNF550 ZNF551 ZNF552 ZNF554 ZNF555 ZNF556 ZNF557
ZNF558 ZNF559 ZNF560 ZNF561 ZNF562 ZNF563 ZNF564 ZNF565 ZNF566 ZNF567
ZNF568 ZNF569 ZNF570 ZNF571 ZNF572 ZNF573 ZNF574 ZNF575 ZNF576 ZNF577
ZNF578 ZNF579 ZNF580 ZNF581 ZNF582 ZNF583 ZNF584 ZNF585A ZNF585B ZNF586
ZNF587 ZNF589 ZNF592 ZNF593 ZNF594 ZNF595 ZNF596 ZNF597 ZNF598 ZNF599
ZNF600 ZNF605 ZNF606 ZNF607 ZNF608 ZNF609 ZNF610 ZNF611 ZNF613 ZNF614
ZNF615 ZNF616 ZNF618 ZNF619 ZNF620 ZNF621 ZNF622 ZNF623 ZNF624 ZNF625
ZNF626 ZNF627 ZNF628 ZNF629 ZNF630 ZNF638 ZNF639 ZNF641 ZNF644 ZNF646
ZNF648 ZNF649 ZNF652 ZNF653 ZNF654 ZNF655 ZNF658 ZNF660 ZNF662 ZNF664
ZNF665 ZNF667 ZNF668 ZNF669 ZNF670 ZNF671 ZNF672 ZNF674 ZNF675 ZNF676
ZNF677 ZNF678 ZNF679 ZNF680 ZNF681 ZNF682 ZNF683 ZNF684 ZNF687 ZNF688
ZNF689 ZNF691 ZNF692 ZNF695 ZNF696 ZNF697 ZNF699 ZNF700 ZNF701 ZNF704
ZNF705A ZNF705B ZNF705D ZNF705G ZNF706 ZNF707 ZNF708 ZNF709 ZNF710 ZNF711
ZNF713 ZNF714 ZNF716 ZNF717 ZNF718 ZNF721 ZNF723 ZNF724 ZNF726 ZNF727
ZNF728 ZNF729 ZNF730 ZNF732 ZNF735 ZNF736 ZNF737 ZNF738 ZNF740 ZNF746
ZNF747 ZNF749 ZNF750 ZNF751 ZNF753 ZNF754 ZNF756 ZNF757 ZNF758 ZNF759
ZNF761 ZNF763 ZNF764 ZNF765 ZNF766 ZNF767P ZNF768 ZNF770 ZNF772 ZNF773
ZNF774 ZNF775 ZNF776 ZNF777 ZNF778 ZNF780A ZNF781 ZNF782 ZNF783 ZNF784
ZNF785 ZNF786 ZNF787 ZNF788P ZNF789 ZNF790 ZNF791 ZNF792 ZNF793 ZNF799
ZNF800 ZNF804A ZNF804B ZNF805 ZNF806 ZNF808 ZNF813 ZNF814 ZNF816 ZNF821
ZNF823 ZNF826P ZNF827 ZNF829 ZNF830 ZNF831 ZNF835 ZNF836 ZNF837 ZNF839
ZNF841 ZNF843 ZNF844 ZNF845 ZNF846 ZNF850 ZNF852 ZNF853 ZNF854 ZNF857
ZNF860 ZNF862 ZNF865 ZNF875 ZNF876P ZNF878 ZNF879 ZNF880 ZNF883 ZNF888
ZNF891 ZNF892 ZNFX1 ZSCAN2 ZSCAN4 ZSCAN9 ZSCAN10 ZSCAN12 ZSCAN16 ZSCAN18
ZSCAN20 ZSCAN21 ZSCAN22 ZSCAN26 ZSCAN29 ZSCAN30 ZSCAN31 ZKSCAN1 ZKSCAN3
ZKSCAN4 ZKSCAN5 ZKSCAN7 ZKSCAN8
""".split())

KNOWN_TFS = {t for t in KNOWN_TFS if t and len(t) > 1}

# -- Config ----------------------------------------------
CELL_TYPES = ['ExcNeuron', 'InhNeuron', 'Astrocyte', 'Oligo', 'OPC', 
              'Microglia', 'Endothelial', 'Pericyte']
CONDITIONS = ['AD', 'Control']
N_TFS = 50
N_TARGETS = 150
THRESHOLD = 0.2
MAX_CELLS = 1000  # Subsample for dCor memory (200×1000×1000×4 = 800MB)

# -- Model loading --------------------------------------
print("Loading BoYue models...")
edge_ckpt = PROJECT_ROOT / "checkpoints" / "main" / "edge_v3_seed0.pt"
dir_ckpt = PROJECT_ROOT / "checkpoints" / "main" / "dir_specialist_tf_non_tf_seed0.pt"

encoder = GraphTransformerEncoderV3(
    G=G, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    sd_prob=0.0).to(device)
edge_head = EdgeHeadV3(d_model=d_model).to(device)

state = torch.load(edge_ckpt, map_location=device, weights_only=True)
encoder.load_state_dict(state['encoder'])
edge_head.load_state_dict(state['edge_head'])
encoder.eval(); edge_head.eval()

# Direction head
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
dir_head.load_state_dict(dstate['dir_head'])
dir_head.eval()
print("  Models loaded.\n")

# -- Load data ------------------------------------------
print("Loading annotated data...")
adata = sc.read_h5ad(DATA_DIR / "processed" / "GSE157827_celltype.h5ad")
print(f"  Shape: {adata.shape}\n")

# -- Functions ------------------------------------------
def compute_P_D(X):
    """Compute Ledoit-Wolf precision (G×G) and gene-gene distance correlation (G×G).
    
    dCor computation: for each gene, compute cell-cell distance matrix, 
    double-center it, then compute pairwise dCov across genes.
    Uses batch processing for memory efficiency.
    """
    n_cells, G = X.shape
    
    # -- P: Ledoit-Wolf precision --
    lw = LedoitWolf().fit(X)
    P = lw.precision_
    np.fill_diagonal(P, 0)
    
    # -- D: Gene-gene distance correlation --
    # Step 1: For each gene, compute double-centered cell-cell distance matrix
    # Store as (G, n_cells, n_cells) in batches
    
    # Compute raw distances per gene (vectorized for genes, allocate in memory)
    # For moderate G and n_cells, this fits in memory
    D_matrices = np.zeros((G, n_cells, n_cells), dtype=np.float32)
    
    for g in range(G):
        x = X[:, g].reshape(-1, 1)
        # Pairwise absolute differences: |x_i - x_j|
        dists = np.abs(x - x.T)  # (n_cells, n_cells)
        # Double-center
        row_mean = dists.mean(axis=1, keepdims=True)
        col_mean = dists.mean(axis=0, keepdims=True)
        grand_mean = dists.mean()
        D_matrices[g] = dists - row_mean - col_mean + grand_mean
    
    # Step 2: Compute dCov and dCor between all gene pairs
    # dCov_sq(i,j) = mean(D_i * D_j)  (element-wise product mean)
    D_flat = D_matrices.reshape(G, -1)  # (G, n_cells*n_cells)
    
    # dCov_sq matrix
    dCov_sq = (D_flat @ D_flat.T) / (n_cells * n_cells)  # (G, G)
    dCov_sq = np.clip(dCov_sq, 0, None)
    
    # dVar (diagonal of dCov_sq)
    dVar = np.diag(dCov_sq).copy()
    dVar[dVar < 1e-12] = 1e-12
    
    # dCor
    D = dCov_sq / np.sqrt(np.outer(dVar, dVar))
    np.fill_diagonal(D, 0)
    D = np.clip(D, -1, 1)
    
    return P, D

@torch.no_grad()
def infer_grn(P, D):
    """Run BoYue inference to get edge probabilities and direction scores."""
    P_t = torch.tensor(P, dtype=torch.float32).unsqueeze(0).to(device)
    D_t = torch.tensor(D, dtype=torch.float32).unsqueeze(0).to(device)
    
    h = encoder(P_t, D_t)
    edge_logits = edge_head(h, P_t, D_t)
    dir_logits = dir_head(h, P_t, D_t)
    
    edge_prob = torch.sigmoid(edge_logits).squeeze(0).cpu().numpy()
    direction = (torch.sigmoid(dir_logits) > 0.5).float().squeeze(0).cpu().numpy()
    dir_score = torch.sigmoid(dir_logits).squeeze(0).cpu().numpy()
    
    # Remove self-loops
    np.fill_diagonal(edge_prob, 0)
    np.fill_diagonal(direction, 0)
    
    return edge_prob, dir_score, direction

# -- Main loop ------------------------------------------
all_results = {}
t_start = time.time()

for ct in CELL_TYPES:
    print(f"\n{'='*60}")
    print(f"{ct}")
    print(f"{'='*60}")
    
    ct_data = adata[adata.obs['cell_type'] == ct].copy()
    n_total = ct_data.shape[0]
    print(f"  Total cells: {n_total:,}")
    
    # Get expression matrix (full genes for gene selection, then subset)
    expr_ct = ct_data.X.toarray() if hasattr(ct_data.X, 'toarray') else ct_data.X
    
    # Compute variance per gene across all cells of this type
    gene_vars = np.var(expr_ct, axis=0)
    
    # Find known TFs in this data
    all_genes = list(ct_data.var_names)
    known_tf_in_data = [(i, g) for i, g in enumerate(all_genes) if g in KNOWN_TFS]
    print(f"  Known TFs in data: {len(known_tf_in_data)}")
    
    # Select top N_TFS TFs by variance
    tf_vars = [(gene_vars[i], g, i) for i, g in known_tf_in_data]
    tf_vars.sort(key=lambda x: -x[0])
    selected_tfs = tf_vars[:N_TFS]
    tf_genes = [t[1] for t in selected_tfs]
    tf_indices = set(t[2] for t in selected_tfs)
    
    # Select N_TARGETS from non-TF genes by variance
    non_tf_candidates = [(gene_vars[i], g, i) for i, g in enumerate(all_genes) 
                         if i not in tf_indices and g not in KNOWN_TFS]
    non_tf_candidates.sort(key=lambda x: -x[0])
    selected_targets = non_tf_candidates[:N_TARGETS]
    
    selected_indices = [t[2] for t in selected_tfs] + [t[2] for t in selected_targets]
    selected_gene_names = [t[1] for t in selected_tfs] + [t[1] for t in selected_targets]
    
    print(f"  Selected: {N_TFS} TFs + {N_TARGETS} targets = {len(selected_indices)} genes")
    print(f"  Example TFs: {tf_genes[:10]}")
    
    # Subset expression to selected genes
    expr_selected = expr_ct[:, selected_indices]
    
    ct_results = {}
    
    for cond in CONDITIONS:
        mask = ct_data.obs['condition'] == cond
        X = expr_selected[mask]
        n_cells = X.shape[0]
        
        print(f"  {cond}: {n_cells:,} cells", end='')
        t0 = time.time()
        
        if n_cells < 100:
            print(" -> SKIP (< 100 cells)")
            continue
        
        # Subsample for speed
        if n_cells > MAX_CELLS:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_cells, MAX_CELLS, replace=False)
            X = X[idx]
            print(f" (subsampled {MAX_CELLS:,})", end='')
        
        P, D = compute_P_D(X)
        edge_prob, dir_score, direction = infer_grn(P, D)
        
        n_edges = int((edge_prob > THRESHOLD).sum())
        elapsed = time.time() - t0
        print(f" -> {n_edges} edges ({elapsed:.1f}s)")
        
        ct_results[cond] = {
            'genes': selected_gene_names,
            'tf_genes': tf_genes,
            'P': P, 'D': D,
            'edge_prob': edge_prob,
            'dir_score': dir_score,
            'direction': direction,
            'n_cells': n_cells
        }
    
    all_results[ct] = {'tf_genes': tf_genes, 'results': ct_results}

# -- Save -----------------------------------------------
out_path = RESULT_DIR / "ad_celltype.pkl"
with open(out_path, 'wb') as f:
    pickle.dump(all_results, f)

total_elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"ALL DONE in {total_elapsed/60:.1f} min")
print(f"Saved to {out_path}")
print(f"{'='*60}")

# -- Quick summary --------------------------------------
print(f"\n{'Cell Type':<15s} {'AD edges':>10s} {'Ctrl edges':>10s} {'Delta':>10s}")
print("-" * 50)
for ct in CELL_TYPES:
    if ct not in all_results: continue
    r = all_results[ct]['results']
    ae = (r['AD']['edge_prob'] > THRESHOLD).sum() if 'AD' in r else 0
    ce = (r['Control']['edge_prob'] > THRESHOLD).sum() if 'Control' in r else 0
    delta = ae - ce
    print(f"  {ct:<15s} {ae:>10,d} {ce:>10,d} {delta:>+10,d}")
