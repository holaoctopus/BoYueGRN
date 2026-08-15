"""Command-line entry points for the DEG-Expanded disease pipeline.

Provides two commands:
    boyue-disease  DEG-Expanded GRN inference (data -> directed GRN CSVs)
    boyue-enrich   GO enrichment on disease output (GRN dir -> GO CSVs)

Default model/TF paths follow the submission layout and can be overridden
with environment variables (BOYUE_CKPT, BOYUE_DATA) or explicit flags.
"""
import argparse
import os
import sys
from pathlib import Path

from .disease import run_disease
from .enrich import run_go


def _default_ckpt_dir():
    env = os.environ.get("BOYUE_CKPT")
    if env:
        return Path(env)
    cwd = Path.cwd()
    for p in (cwd / "checkpoints", cwd.parent / "checkpoints"):
        if (p / "main").exists():
            return p
    return None


def _default_data_root():
    env = os.environ.get("BOYUE_DATA")
    if env:
        return Path(env)
    cwd = Path.cwd()
    for p in (cwd / "data_external", cwd.parent / "data_external"):
        if p.exists():
            return p
    return None


def _resolve_ckpt(flag, name, ckpt_dir):
    if flag:
        return Path(flag)
    if ckpt_dir is not None:
        p = ckpt_dir / "main" / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Checkpoint not found: {name}. Pass --ckpt-dir or set BOYUE_CKPT, "
        "or use --edge-ckpt / --dir-ckpt explicitly.")


def _resolve_tf_csv(flag):
    if flag:
        return Path(flag)
    root = _default_data_root()
    if root is not None:
        p = root / "BEELINE" / "human-tfs.csv"
        if p.exists():
            return p
    raise FileNotFoundError(
        "TF list not found. Pass --tf-list or set BOYUE_DATA with a "
        "BEELINE/human-tfs.csv inside.")


def _add_common_model_args(p):
    p.add_argument("--edge-ckpt", default=None, help="edge_v3 checkpoint (.pt)")
    p.add_argument("--dir-ckpt", default=None, help="TT direction specialist checkpoint (.pt)")
    p.add_argument("--ckpt-dir", default=None,
                   help="directory containing main/edge_v3_seed0.pt etc. "
                        "(default: $BOYUE_CKPT or ./checkpoints)")
    p.add_argument("--tf-list", default=None,
                   help="one-column CSV of known TF symbols "
                        "(default: $BOYUE_DATA/BEELINE/human-tfs.csv)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])


def main_disease(argv=None):
    p = argparse.ArgumentParser(
        prog="boyue-disease",
        description="DEG-Expanded directed GRN inference from scRNA-seq.")
    p.add_argument("--data", required=True,
                   help="h5ad file, or directory with count.txt.gz + "
                        "metadata.txt.gz")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--celltype-key", default="cell_type",
                   help="obs column for cell type (h5ad mode)")
    p.add_argument("--stage-key", default="condition",
                   help="obs column for condition/stage (h5ad mode)")
    p.add_argument("--baseline", required=True, help="baseline condition")
    p.add_argument("--stages", default=None, nargs="+",
                   help="conditions to analyze (default: all non-baseline)")
    p.add_argument("--case-label", default="case",
                   help="label used in outputs (e.g. 'NAFLD')")
    p.add_argument("--no-normalize", action="store_true",
                   help="skip QC + normalize_total + log1p (data already "
                        "log-normalized)")
    p.add_argument("--min-cells-per-cond", type=int, default=100)
    p.add_argument("--min-cells-deg", type=int, default=50)
    p.add_argument("--min-deg", type=int, default=50)
    p.add_argument("--fdr", type=float, default=0.05, dest="fdr_thr")
    p.add_argument("--l2fc", type=float, default=0.5, dest="l2fc_thr")
    p.add_argument("--edge-threshold", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=50,
                   help="pre-screen top-k candidate targets per TF")
    p.add_argument("--max-cells", type=int, default=500,
                   help="cell subsample for window inference")
    p.add_argument("--max-cells-input", type=int, default=1000,
                   help="cell subsample per cell type x stage before inference")
    p.add_argument("--max-tfs", type=int, default=150,
                   help="cap on number of TFs per cell type (None = no cap)")
    _add_common_model_args(p)
    args = p.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else _default_ckpt_dir()
    edge_ckpt = _resolve_ckpt(args.edge_ckpt, "edge_v3_seed0.pt", ckpt_dir)
    dir_ckpt = _resolve_ckpt(args.dir_ckpt, "dir_specialist_tf_tf_seed0.pt",
                             ckpt_dir)
    tf_csv = _resolve_tf_csv(args.tf_list)

    device = None if args.device == "auto" else args.device
    run_disease(
        args.data, args.out,
        celltype_key=args.celltype_key,
        stage_key=args.stage_key,
        baseline=args.baseline,
        stages=args.stages,
        tf_csv=tf_csv,
        edge_ckpt=edge_ckpt,
        dir_ckpt=dir_ckpt,
        device=device,
        normalize=not args.no_normalize,
        min_cells_per_cond=args.min_cells_per_cond,
        min_cells_deg=args.min_cells_deg,
        min_deg=args.min_deg,
        fdr_thr=args.fdr_thr,
        l2fc_thr=args.l2fc_thr,
        edge_threshold=args.edge_threshold,
        top_k=args.top_k,
        max_cells=args.max_cells,
        max_cells_input=args.max_cells_input,
        max_tfs=args.max_tfs,
        case_label=args.case_label,
        verbose=True,
    )


def main_enrich(argv=None):
    p = argparse.ArgumentParser(
        prog="boyue-enrich",
        description="GO enrichment on DEG-Expanded GRN output.")
    p.add_argument("--in", dest="grn_dir", required=True,
                   help="out_dir/grn produced by boyue-disease")
    p.add_argument("--out", required=True, help="output directory for GO CSVs")
    p.add_argument("--case-label", default="case")
    p.add_argument("--stages", default=None, nargs="+",
                   help="stage names to include (default: all present)")
    p.add_argument("--go-lib", default="GO_Biological_Process_2023",
                   help="gseapy gene set library")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--permutation-num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-size", type=int, default=5)
    p.add_argument("--max-size", type=int, default=500)
    args = p.parse_args(argv)

    run_go(args.grn_dir, args.out, case_label=args.case_label,
           stages=args.stages, go_lib=args.go_lib, threads=args.threads,
           permutation_num=args.permutation_num, seed=args.seed,
           min_size=args.min_size, max_size=args.max_size, verbose=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("disease", "enrich"):
        fn = main_disease if sys.argv[1] == "disease" else main_enrich
        sys.exit(fn(sys.argv[2:]))
    print("Usage: boyue disease|enrich ...")
    sys.exit(1)
