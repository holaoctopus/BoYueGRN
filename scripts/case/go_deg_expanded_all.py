#!/usr/bin/env python
"""GO enrichment for DEG-Expanded results: AD / Periodontitis / HCC / PVL.

Usage:
  python scripts/case/go_deg_expanded_all.py <case>
  case ∈ {ad, perio, hcc, pvl}

For each cell type × stage, builds a ranked target list (max edge_prob over TFs,
excluding ribo/generic), runs GSEA prerank (GO_Biological_Process_2023), saves CSV.
"""
import sys, pickle, time, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import gseapy as gp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _config import PROJECT_ROOT as PROJECT, RESULT_ROOT

# Per-case configuration: (result_dir, out_dir, stages, case_label)
CASE_CONFIG = {
    'ad':    ('results/4_case_studies/ad/deg_expanded',
              'results/4_case_studies/ad/go_deg_expanded',
              ['control', 'ad'],
              'AD'),
    'perio': ('results/4_case_studies/periodontitis/deg_expanded',
              'results/4_case_studies/periodontitis/go_deg_expanded',
              ['bm', 'gm', 'pd'],
              'Periodontitis'),
    'hcc':   ('results/4_case_studies/hcc/deg_expanded',
              'results/4_case_studies/hcc/go_deg_expanded',
              ['normal', 'tumor'],
              'HCC'),
    'pvl':   ('results/4_case_studies/pvl/deg_expanded',
              'results/4_case_studies/pvl/go_deg_expanded',
              ['healthy', 'a', 'b', 'c'],
              'PVL'),
}

RIBO_PREFIXES = ('RPL', 'RPS', 'MRPL', 'MRPS')
GENERIC_TF_TERMS = {
    'Regulation Of DNA-templated Transcription (GO:0006355)',
    'Positive Regulation Of DNA-templated Transcription (GO:0045893)',
    'Negative Regulation Of DNA-templated Transcription (GO:0045892)',
    'Regulation Of Transcription By RNA Polymerase II (GO:0006357)',
    'Positive Regulation Of Transcription By RNA Polymerase II (GO:0045944)',
    'Negative Regulation Of Transcription By RNA Polymerase II (GO:0045893)',
    'Transcription By RNA Polymerase II (GO:0006366)',
    'DNA-templated Transcription (GO:0006351)',
    'Positive Regulation Of Gene Expression (GO:0010628)',
    'Negative Regulation Of Gene Expression (GO:0010629)',
}
GO_LIB = 'GO_Biological_Process_2023'

def is_generic(term):
    kw = ['translat', 'ribosom', 'mitochondri', 'oxidative phosphorylat',
          'respiratory', 'mrna splic', 'trna', 'electron transport']
    return any(k in term.lower() for k in kw)

def build_ranked_targets(ep_sparse, genes, tf_genes):
    """Max edge_prob over TFs for each target (TF->target). Excludes ribo/etc."""
    tf_set = set(str(t).upper() for t in tf_genes)
    tf_idx = [i for i, g in enumerate(genes) if g.upper() in tf_set]
    target_idx = [j for j in range(len(genes)) if j not in set(tf_idx)]
    if not tf_idx or not target_idx:
        return pd.Series(dtype=float)
    ep_block = ep_sparse[np.ix_(tf_idx, target_idx)].toarray()
    max_prob = ep_block.max(axis=0)
    target_genes = [genes[j] for j in target_idx]
    keep = [(g, float(m)) for g, m in zip(target_genes, max_prob)
            if not any(g.upper().startswith(p) for p in RIBO_PREFIXES)
            and not g.upper().startswith(('ENSG', 'LINC', 'MIR', 'AC0', 'AL0'))]
    return pd.Series(dict(keep)).sort_values(ascending=False)

def run_prerank(rnk, case, stage):
    if len(rnk) < 15:
        return pd.DataFrame()
    try:
        res = gp.prerank(rnk=rnk, gene_sets=GO_LIB, organism='human',
                         outdir=None, min_size=5, max_size=500,
                         permutation_num=1000, seed=42, threads=2, silent=True)
        df = res.res2d.copy()
        if len(df) == 0:
            return df
        df['Term'] = df['Term'].str.split('~').str[-1].str.strip()
        df['stage'] = stage; df['case'] = case
        return df[['case', 'stage', 'Term', 'NES', 'NOM p-val', 'FDR q-val']]
    except Exception as e:
        print(f"    [WARN] {case}/{stage}: {e}")
        return pd.DataFrame()

def main(case_key):
    if case_key not in CASE_CONFIG:
        print(f"Unknown case: {case_key}. Use one of {list(CASE_CONFIG.keys())}")
        sys.exit(1)
    res_subdir, out_subdir, stages, case_label = CASE_CONFIG[case_key]
    EXP_DIR = PROJECT / res_subdir
    OUT_DIR = PROJECT / out_subdir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find cell-type subdirectories
    ct_dirs = [d for d in EXP_DIR.iterdir() if d.is_dir() and (d / "metadata.pkl").exists()]
    if not ct_dirs:
        print(f"No cell-type dirs with metadata.pkl in {EXP_DIR}")
        sys.exit(1)
    print(f"{'='*60}")
    print(f"GO Enrichment: {case_label} ({case_key})")
    print(f"  Cell types: {[d.name for d in ct_dirs]}")
    print(f"  Stages: {stages}")
    print(f"{'='*60}")

    all_rows = []
    for ct_dir in sorted(ct_dirs):
        with open(ct_dir / "metadata.pkl", 'rb') as f:
            meta = pickle.load(f)
        tf_genes = meta['tf_genes']
        all_deg_genes = meta['all_deg_genes']
        ct_name = ct_dir.name
        print(f"\n[{ct_name}] DEGs={len(all_deg_genes)} ({len(tf_genes)} TFs)")

        for stage in stages:
            pkl_path = ct_dir / f"{stage}.pkl"
            if not pkl_path.exists():
                print(f"  {stage}: [SKIP] no pkl"); continue
            with open(pkl_path, 'rb') as f:
                sd = pickle.load(f)
            ep_sparse = sd['edge_prob_sparse']
            n_edges = sd['n_edges_above_thresh']
            rnk = build_ranked_targets(ep_sparse, all_deg_genes, tf_genes)
            print(f"  {stage}: {len(rnk)} targets, {n_edges:,} edges", end='')
            t0 = time.time()
            df = run_prerank(rnk, case_label, stage)
            print(f" -> {len(df)} terms ({time.time()-t0:.0f}s)")
            if len(df) > 0:
                df['cell_type'] = ct_name
                all_rows.append(df)
            del ep_sparse; gc.collect()

    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        big.to_csv(OUT_DIR / f"{case_label}_stage_go_raw.csv", index=False)
        big_f = big[~big['Term'].isin(GENERIC_TF_TERMS)].copy()
        big_f.to_csv(OUT_DIR / f"{case_label}_stage_go.csv", index=False)
        print(f"\n  Total: {len(big_f)} rows, {big_f['Term'].nunique()} unique terms")
        print(f"  Saved -> {OUT_DIR / (case_label + '_stage_go.csv')}")
    else:
        print("  [ERROR] No results")
    print(f"\nDONE: {case_key}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/case/go_deg_expanded_all.py <case>  (ad|perio|hcc|pvl)")
        sys.exit(1)
    main(sys.argv[1])
