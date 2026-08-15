"""GO enrichment for DEG-Expanded GRN results (Result 3 of the paper).

Consumes the output of boyue.disease.run_disease (out_dir/grn layout):
    grn/{ct}/genes.json              DEG gene set + TF list
    grn/{ct}/{stage}_edge_prob.npz   sparse edge-prob matrix (n_deg x n_deg)

For each cell type x stage it builds a ranked target list (max edge_prob
over TFs for every non-TF target, excluding ribo/generic genes), runs
GSEA prerank (GO_Biological_Process_2023 by default) and saves one CSV
with the raw results and one with generic transcription terms removed.

Output:
    out_dir/{case_label}_stage_go.csv        filtered GO terms
    out_dir/{case_label}_stage_go_raw.csv    all GO terms
"""
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gseapy as gp
from scipy import sparse as sp_sparse

warnings.filterwarnings("ignore")

RIBO_PREFIXES = ("RPL", "RPS", "MRPL", "MRPS")
GENERIC_TF_TERMS = {
    "Regulation Of DNA-templated Transcription (GO:0006355)",
    "Positive Regulation Of DNA-templated Transcription (GO:0045893)",
    "Negative Regulation Of DNA-templated Transcription (GO:0045892)",
    "Regulation Of Transcription By RNA Polymerase II (GO:0006357)",
    "Positive Regulation Of Transcription By RNA Polymerase II (GO:0045944)",
    "Negative Regulation Of Transcription By RNA Polymerase II (GO:0045893)",
    "Transcription By RNA Polymerase II (GO:0006366)",
    "DNA-templated Transcription (GO:0006351)",
    "Positive Regulation Of Gene Expression (GO:0010628)",
    "Negative Regulation Of Gene Expression (GO:0010629)",
}
DEFAULT_GO_LIB = "GO_Biological_Process_2023"


def is_generic(term):
    kw = ["translat", "ribosom", "mitochondri", "oxidative phosphorylat",
          "respiratory", "mrna splic", "trna", "electron transport"]
    return any(k in term.lower() for k in kw)


def build_ranked_targets(ep_sparse, genes, tf_genes):
    """Max edge_prob over TFs for each non-TF target. Excludes ribo/etc."""
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
            and not g.upper().startswith(("ENSG", "LINC", "MIR", "AC0", "AL0"))]
    return pd.Series(dict(keep)).sort_values(ascending=False)


def run_prerank(rnk, case, stage, go_lib=DEFAULT_GO_LIB, outdir=None,
                threads=2, permutation_num=1000, seed=42, min_size=5,
                max_size=500, verbose=False):
    """GSEA prerank on a ranked gene list. Returns a DataFrame (may be empty)."""
    if len(rnk) < 15:
        return pd.DataFrame()
    try:
        res = gp.prerank(rnk=rnk, gene_sets=go_lib, organism="human",
                         outdir=outdir, min_size=min_size, max_size=max_size,
                         permutation_num=permutation_num, seed=seed,
                         threads=threads, silent=not verbose)
        df = res.res2d.copy()
        if len(df) == 0:
            return df
        df["Term"] = df["Term"].str.split("~").str[-1].str.strip()
        df["stage"] = stage; df["case"] = case
        return df[["case", "stage", "Term", "NES", "NOM p-val", "FDR q-val"]]
    except Exception as e:
        if verbose:
            print(f"    [WARN] {case}/{stage}: {e}")
        return pd.DataFrame()


def run_go(grn_dir, out_dir, *, case_label="case", stages=None,
           go_lib=DEFAULT_GO_LIB, threads=2, permutation_num=1000,
           seed=42, min_size=5, max_size=500, verbose=True):
    """Run GO enrichment over all cell types x stages under grn_dir.

    grn_dir: the out_dir/grn directory produced by boyue.disease.run_disease.
    stages:  optional list of stage names to include (default: discover from
             *_edge_prob.npz files under each cell-type directory).
    """
    grn_dir = Path(grn_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ct_dirs = [d for d in grn_dir.iterdir()
               if d.is_dir() and (d / "genes.json").exists()]
    if not ct_dirs:
        raise FileNotFoundError(
            f"No cell-type dirs with genes.json under {grn_dir}; run "
            "boyue.disease first")

    if verbose:
        print(f"GO Enrichment: {case_label}")
        print(f"  Cell types: {[d.name for d in ct_dirs]}")
        print(f"  GO library: {go_lib}")

    all_rows = []
    for ct_dir in sorted(ct_dirs):
        with open(ct_dir / "genes.json") as f:
            meta = json.load(f)
        tf_genes = meta["tf_genes"]
        all_deg_genes = meta["all_deg_genes"]
        ct_name = meta.get("cell_type", ct_dir.name)
        if verbose:
            print(f"\n[{ct_name}] DEGs={len(all_deg_genes)} "
                  f"({len(tf_genes)} TFs)")

        npz_files = sorted(ct_dir.glob("*_edge_prob.npz"))
        if stages is None:
            stage_names = [p.name.replace("_edge_prob.npz", "") for p in npz_files]
        else:
            stage_names = [s for s in stages
                           if (ct_dir / f"{s}_edge_prob.npz").exists()]
        for stage in stage_names:
            npz_path = ct_dir / f"{stage}_edge_prob.npz"
            if not npz_path.exists():
                if verbose:
                    print(f"  {stage}: [SKIP] no npz")
                continue
            ep_sparse = sp_sparse.load_npz(npz_path).tocsr()
            rnk = build_ranked_targets(ep_sparse, all_deg_genes, tf_genes)
            if verbose:
                print(f"  {stage}: {len(rnk)} targets", end="", flush=True)
            t0 = time.time()
            df = run_prerank(rnk, case_label, stage, go_lib=go_lib,
                             threads=threads, permutation_num=permutation_num,
                             seed=seed, min_size=min_size, max_size=max_size,
                             verbose=verbose)
            if verbose:
                print(f" -> {len(df)} terms ({time.time() - t0:.0f}s)")
            if len(df) > 0:
                df["cell_type"] = ct_name
                all_rows.append(df)
            del ep_sparse; gc.collect()

    if not all_rows:
        print(f"[WARN] No GO terms were produced for {case_label}. "
              "Check that gene symbols in the data are mappable to the GO "
              f"library ({go_lib}). Saving empty outputs.")
        empty = pd.DataFrame(columns=["case", "stage", "Term", "NES",
                                      "NOM p-val", "FDR q-val", "cell_type"])
        empty.to_csv(out_dir / f"{case_label}_stage_go_raw.csv", index=False)
        empty.to_csv(out_dir / f"{case_label}_stage_go.csv", index=False)
        return empty

    big = pd.concat(all_rows, ignore_index=True)
    big.to_csv(out_dir / f"{case_label}_stage_go_raw.csv", index=False)
    big_f = big[~big["Term"].isin(GENERIC_TF_TERMS)].copy()
    big_f.to_csv(out_dir / f"{case_label}_stage_go.csv", index=False)
    if verbose:
        print(f"\n  Total: {len(big_f)} rows, "
              f"{big_f['Term'].nunique()} unique terms")
        print(f"  Saved -> {out_dir / (case_label + '_stage_go.csv')}")
    return big_f
