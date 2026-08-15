"""BoYue path configuration.

All scripts import paths from here. Users can override defaults via
environment variables:

    BOYUE_ROOT    - repository root (default: auto-detected)
    BOYUE_DATA    - directory containing downloaded raw datasets
                    (default: $BOYUE_ROOT/data_external/)
    BOYUE_CKPT    - directory containing model checkpoints
                    (default: $BOYUE_ROOT/checkpoints/)

Typical setup after downloading raw data:
    export BOYUE_DATA=/path/to/downloaded_datasets
    python scripts/case/eval_zhongliang_celltype.py
"""
import os
from pathlib import Path

# Repository root: this file is at <repo>/scripts/_config.py
SUBMISSION_ROOT = Path(__file__).resolve().parent.parent

# Allow env override (useful for read-only installs)
PROJECT_ROOT = Path(os.environ.get("BOYUE_ROOT", SUBMISSION_ROOT))

# Data root: where users place downloaded raw datasets
DATA_ROOT = Path(os.environ.get("BOYUE_DATA", PROJECT_ROOT / "data_external"))

# Checkpoint root
CKPT_ROOT = Path(os.environ.get("BOYUE_CKPT", PROJECT_ROOT / "checkpoints"))

# Main model checkpoints
# - edge_v3: 4-seed ensemble (edge existence)
# - dir_specialist_tf_non_tf: 4-seed ensemble (NT specialist, used in BEELINE benchmark)
# - dir_specialist_tf_tf: seed_0 only (TT specialist, best direction accuracy,
#   used in DEG-Expanded case studies and genome-wide pipeline)
EDGE_CKPTS = [CKPT_ROOT / "main" / f"edge_v3_seed{s}.pt" for s in range(4)]
DIR_CKPTS = [CKPT_ROOT / "main" / f"dir_specialist_tf_non_tf_seed{s}.pt" for s in range(4)]
TT_DIR_CKPT = CKPT_ROOT / "main" / "dir_specialist_tf_tf_seed0.pt"

# Results root
RESULT_ROOT = PROJECT_ROOT / "results"


def data_dir(dataset):
    """Return path to a dataset directory."""
    return DATA_ROOT / dataset


def result_dir(subdir):
    """Return path to a results subdirectory (creates if missing)."""
    p = RESULT_ROOT / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p
