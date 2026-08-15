"""BoYue: Directed GRN inference from scRNA-seq via Graph Transformer.

Core package providing:
- model: GraphTransformerEncoderV3 + EdgeHeadV3 + AsymmetricDirHead
- stats: Ledoit-Wolf precision (P) + distance correlation (D)
- data: PDNDataset for training
- infer: high-level inference API (load checkpoints -> directed GRN)
"""
from .model import (
    GraphTransformerEncoderV3,
    EdgeHeadV3,
    AsymmetricDirHead,
    DEFAULT_CONFIG,
)
from .stats import compute_P, compute_D, compute_PDN
from .infer import BoYueInferencer, load_ensemble
from .genome_wide import GenomeWideInferencer
from .disease import (
    run_disease,
    compute_degs,
    expanded_deg_grn,
    read_expression,
    load_known_tfs,
    load_models,
)
from .enrich import run_go

__version__ = "1.0.0"

__all__ = [
    "GraphTransformerEncoderV3",
    "EdgeHeadV3",
    "AsymmetricDirHead",
    "DEFAULT_CONFIG",
    "compute_P",
    "compute_D",
    "compute_PDN",
    "BoYueInferencer",
    "load_ensemble",
    "GenomeWideInferencer",
    "run_disease",
    "compute_degs",
    "expanded_deg_grn",
    "read_expression",
    "load_known_tfs",
    "load_models",
    "run_go",
]
