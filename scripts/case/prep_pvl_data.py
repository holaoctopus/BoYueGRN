#!/usr/bin/env python
"""PVL prep: build full-gene h5ad + extract Periodontitis BM healthy cells.

Reuses the exact annotate_zhongliang pipeline (read_10x_mtx + concat index_unique='-')
but SKIPS the HVG subsetting (which dropped 33538 -> 3000 genes), so DEG-Expanded
has full genes to work with. Cell types are re-annotated with the same markers.

Output (CPU-only, runs while GPU busy with AD):
  results/4_case_studies/pvl/deg_expanded/pvl_fullgene.h5ad      (PVL full-gene, A/B/C)
  results/4_case_studies/pvl/deg_expanded/bm_healthy_cells.h5ad  (Periodontitis BM, healthy oral)

Both are log-normalized (normalize_total 1e4 + log1p), gene-symbol indexed.
The PVL DEG-Expanded script loads both, intersects genes, uses BM as baseline.
"""
import sys, gzip, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, DATA_ROOT, RESULT_ROOT
PVL_RAW = DATA_ROOT / "zhongliang" / "GSE196296"
PERIO_MTX = DATA_ROOT / "GSE164241pero" / "mtx" / "count"
OUT_DIR = RESULT_ROOT / "4_case_studies" / "pvl" / "deg_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

t_start = time.time()

# Oral mucosa markers (same as annotate_zhongliang + eval_periodontitis)
MARKERS = {
    'Epithelial':  ['KRT5','KRT14','KRT13','KRT4','TP63','CDH1','EPCAM','KRT15','KRT19','SFN','CLDN1','CLDN4','DSP','PKP1'],
    'Fibroblast':  ['COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA','FAP','COL6A1','COL6A2','THY1','S100A4','SPARC','LAMA4'],
    'T_cell':      ['CD3D','CD3E','CD3G','CD4','CD8A','CD8B','TRAC','TRBC1','TRBC2','CD2','CCL5','NKG7','GZMA','PRF1'],
    'B_Plasma':    ['CD19','MS4A1','CD79A','CD79B','JCHAIN','MZB1','SDC1','XBP1','IGHG1','IGHA1','IGKC','IGLC2'],
    'Myeloid':     ['CD14','CD68','CD163','CSF1R','ITGAM','LYZ','CST3','AIF1','ITGAX','FCGR3A','FCER1G','TYROBP'],
    'Mast':        ['KIT','TPSAB1','CPA3','HDC','MS4A2','GATA2','TPSB2'],
    'Endothelial': ['PECAM1','VWF','CDH5','ENG','ESAM','CLDN5','FLT1','KDR'],
    'Myocyte':     ['ACTA2','MYH11','CNN1','TAGLN','MYLK','ACTG2','DES','CALD1'],
}

def annotate_celltypes(adata):
    """Marker-based cell type scoring (same as annotate_zhongliang)."""
    expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    var_list = list(adata.var_names)
    scores = {}
    for ct, mgs in MARKERS.items():
        present = [g for g in mgs if g in var_list]
        if len(present) < 2:
            scores[ct] = np.zeros(adata.shape[0]); continue
        gene_idx = [var_list.index(g) for g in present]
        scores[ct] = np.asarray(expr[:, gene_idx].mean(axis=1)).ravel()
    score_matrix = np.column_stack([scores[ct] for ct in MARKERS.keys()])
    best_idx = np.argmax(score_matrix, axis=1)
    ct_names = list(MARKERS.keys())
    adata.obs['cell_type'] = [ct_names[i] for i in best_idx]
    max_scores = np.max(score_matrix, axis=1)
    adata.obs.loc[max_scores < 0.1, 'cell_type'] = 'Unknown'
    return adata

# ===========================================================
# 1. Build PVL full-gene h5ad (read_10x_mtx + concat, NO HVG)
# ===========================================================
print("=" * 60)
print("[1/3] Building PVL full-gene h5ad (NO HVG subset)")
print("=" * 60)

SAMPLES = {'A': 'GSM5870060_LesionA',
           'B': 'GSM5870062_LesionB',
           'C': 'GSM5870063_LesionC'}

adatas = {}
for label, prefix in SAMPLES.items():
    print(f"  Lesion {label}: {prefix}")
    a = sc.read_10x_mtx(str(PVL_RAW), prefix=f"{prefix}_", gex_only=False)
    a.obs['sample'] = label
    a.obs_names = [f"{label}_{bc}" for bc in a.obs_names]
    adatas[label] = a
    print(f"    {a.shape[0]:,} cells x {a.shape[1]:,} genes")

adata_pvl = anndata.concat([adatas['A'], adatas['B'], adatas['C']],
                            label='sample', keys=['A', 'B', 'C'], index_unique='-')
del adatas; gc.collect()
print(f"  Combined: {adata_pvl.shape}")

# QC (same as annotate script)
adata_pvl.var['mt'] = adata_pvl.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata_pvl, qc_vars=['mt'], inplace=True)
sc.pp.filter_cells(adata_pvl, min_genes=200)
sc.pp.filter_genes(adata_pvl, min_cells=3)
adata_pvl = adata_pvl[adata_pvl.obs['n_genes_by_counts'] < 6000, :].copy()
adata_pvl = adata_pvl[adata_pvl.obs['pct_counts_mt'] < 20, :].copy()
print(f"  After QC: {adata_pvl.shape}")

# Normalize (NO HVG)
sc.pp.normalize_total(adata_pvl, target_sum=1e4)
sc.pp.log1p(adata_pvl)

# Annotate cell types
adata_pvl = annotate_celltypes(adata_pvl)
print(f"  cell_type: {adata_pvl.obs['cell_type'].value_counts().to_dict()}")
print(f"  sample: {adata_pvl.obs['sample'].value_counts().to_dict()}")

adata_pvl.write_h5ad(OUT_DIR / "pvl_fullgene.h5ad")
print(f"  Saved -> pvl_fullgene.h5ad")

# ===========================================================
# 2. Extract Periodontitis BM (healthy) cells
# ===========================================================
print(f"\n{'='*60}")
print("[2/3] Extracting Periodontitis BM (healthy oral) cells")
print("=" * 60)

genes = []
with gzip.open(PERIO_MTX / "features.tsv.gz", 'rt') as f:
    for line in f:
        cols = line.strip().split('\t')
        genes.append(cols[0])
print(f"  {len(genes):,} genes")

barcodes = []
with gzip.open(PERIO_MTX / "barcodes.tsv.gz", 'rt') as f:
    for line in f:
        barcodes.append(line.strip())

cond_map = {}
for bc in barcodes:
    parts = bc.split('_')
    donor = parts[1] if len(parts) > 1 else parts[0]
    if donor.startswith('BM'):
        cond = 'BM'
    elif donor.startswith('GM'):
        cond = 'GM'
    elif donor.startswith('PD'):
        cond = 'PD'
    else:
        cond = 'PD'
    cond_map[bc] = cond
print(f"  Condition: {pd.Series(list(cond_map.values())).value_counts().to_dict()}")

print("  Loading sparse matrix...")
t0 = time.time()
rows, cols, vals = [], [], []
with gzip.open(PERIO_MTX / "matrix.mtx.gz", 'rt') as f:
    header = f.readline()
    while header.startswith('%'):
        header = f.readline()
    n_genes, n_cells, n_entries = map(int, header.strip().split())
    print(f"  Matrix: {n_genes} x {n_cells}, {n_entries:,} entries")
    for line in f:
        g, c, v = line.strip().split()
        rows.append(int(g) - 1); cols.append(int(c) - 1); vals.append(float(v))
data_sparse = sparse.coo_matrix((vals, (rows, cols)),
                                 shape=(n_genes, n_cells)).tocsc().T.tocsr()
del rows, cols, vals; gc.collect()
print(f"  Loaded in {time.time()-t0:.0f}s, shape: {data_sparse.shape}")

obs = pd.DataFrame({'barcode': barcodes,
                    'condition': [cond_map[bc] for bc in barcodes]}, index=barcodes)
var = pd.DataFrame({'gene': genes}, index=genes)
adata_perio = sc.AnnData(X=data_sparse, obs=obs, var=var)
sc.pp.filter_cells(adata_perio, min_genes=200)
sc.pp.filter_genes(adata_perio, min_cells=3)
sc.pp.normalize_total(adata_perio, target_sum=1e4)
sc.pp.log1p(adata_perio)

# Annotate cell types on full Periodontitis data (so BM has cell_type labels)
adata_perio = annotate_celltypes(adata_perio)
print(f"  Periodontitis cell_type: {adata_perio.obs['cell_type'].value_counts().to_dict()}")

bm = adata_perio[adata_perio.obs['condition'] == 'BM'].copy()
print(f"  BM healthy cells: {bm.shape}")
print(f"  BM cell_type: {bm.obs['cell_type'].value_counts().to_dict()}")
bm.write_h5ad(OUT_DIR / "bm_healthy_cells.h5ad")
print(f"  Saved -> bm_healthy_cells.h5ad")

# ===========================================================
# 3. Gene overlap summary
# ===========================================================
print(f"\n{'='*60}")
print("[3/3] Gene overlap summary")
print("=" * 60)
pvl_genes = set(adata_pvl.var_names)
bm_genes = set(bm.var_names)
shared = pvl_genes & bm_genes
print(f"  PVL genes: {len(pvl_genes):,}")
print(f"  BM genes:  {len(bm_genes):,}")
print(f"  Shared:    {len(shared):,} ({100*len(shared)/len(pvl_genes):.1f}% of PVL)")

# Cell type overlap
pvl_cts = set(adata_pvl.obs['cell_type'].unique())
bm_cts = set(bm.obs['cell_type'].unique())
print(f"  PVL cell types: {sorted(pvl_cts)}")
print(f"  BM cell types:  {sorted(bm_cts)}")
print(f"  Shared cell types: {sorted(pvl_cts & bm_cts)}")

print(f"\nDONE in {(time.time()-t_start)/60:.1f} min")
