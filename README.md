# BoYueGRN: Directed Gene Regulatory Network Inference from scRNA-seq

Code and data accompanying the manuscript:
**"BoYueGRN: Zero-shot causal discovery of directed gene regulatory networks
from single-cell transcriptomes via amortized inference over synthetic
structural causal models"**

> 中文版说明见 [README_zh.md](README_zh.md)。

BoYueGRN is an amortized causal-discovery framework trained exclusively on
synthetic structural causal models. A single forward pass infers both edge
probabilities and regulatory directions for any new single-cell dataset, and
TF-centric sliding windows with asymmetric fusion extend the fixed-size model
to whole-transcriptome coverage.

**About the name.** "BoYue" (博约, *bó yuē*) is taken from Su Shi's (苏轼,
Song dynasty) "博观而约取，厚积而薄发" — "observe broadly and take precisely,
accumulate richly and release sparingly". It reflects the amortized paradigm
of this framework: the model observes a broad ensemble of synthetic SCMs
offline, then amortizes that cost into a single forward pass on any new
dataset.

This repository contains the **model code and the full training / benchmark /
case-study pipeline**, with repository-relative paths (no machine-specific
absolute paths). Model weights and the large external datasets (BEELINE /
Perturb-seq h5ad / SCM cache / DEG-expanded pkls) are **not** bundled; how to
point the pipeline at them is described in §3.

***

## 1. Repository layout

```
BoYueGRN/
├── README.md                       # this file
├── README_zh.md                    # Chinese version of this file
├── LICENSE                         # Apache License 2.0
├── requirements.txt                # base Python dependencies
├── requirements-pipeline.txt       # full-pipeline deps (torch, sklearn, ...)
├── pyproject.toml                  # pip-installable package (boyue)
├── boyue/                          # BoYueGRN model/inference package
│   ├── model.py                    #   GraphTransformerEncoderV3 + edge/dir heads
│   ├── stats.py                    #   compute_P (Ledoit-Wolf) / compute_D / compute_N
│   ├── data.py                     #   PDNDataset (training data loader)
│   ├── infer.py                    #   load_ensemble / predict_from_expression / directed_edges
│   ├── genome_wide.py              #   whole-genome sliding-window pipeline
│   ├── disease.py                  #   disease-case inference + DEG computation
│   ├── enrich.py                   #   GO enrichment on disease GRN outputs
│   ├── cli.py                      #   `boyue-disease` / `boyue-enrich` entry points
│   └── __init__.py                 #   public API + __version__ = "1.0.0"
└── scripts/                        # full pipeline (no figure scripts)
    ├── train/                      # model training (edge v3, dir specialists)
    ├── eval/                       # benchmark / cross-specialist evaluation
    ├── case/                       # 5 disease case-study inference + GO
    ├── data_gen/                   # synthetic SCM cache generators (P/A, P/D/N)
    └── _config.py                  # env-overridable path config (pipeline)
```

## 2. The boyue package (model code)

```bash
pip install -e .                     # or: pip install -r requirements-pipeline.txt

# quick sanity check on a random 200x200 expression matrix (CPU)
python -c "from boyue import load_ensemble; import numpy as np, torch; \
inf = load_ensemble(edge_ckpts=['checkpoints/main/edge_v3_seed0.pt'], \
dir_ckpts=['checkpoints/main/dir_specialist_tf_non_tf_seed0.pt'], device=torch.device('cpu')); \
print(inf.predict_from_expression(np.random.gamma(2.0, 1.0, (200, 200)).astype(np.float32)).keys())"
```

Public API:

- `load_ensemble` — load trained checkpoints into an inference object.
- `predict_from_expression` — infer edge probabilities + directions from an
  expression matrix.
- `directed_edges` — return directed edges as a DataFrame.
- `GenomeWideInferencer` — sliding-window whole-genome inference.
- `run_disease` / `compute_degs` / `expanded_deg_grn` — case-study pipeline.
- `run_go` — GO enrichment on case-study GRN outputs.

## 3. Full pipeline (training / evaluation / inference)

All pipeline code uses repository-relative paths; every external path is
overridable via environment variables (defined in `scripts/_config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `BOYUE_ROOT` | repo root (auto-detected) | workspace root containing `data_external/`, `checkpoints/` |
| `BOYUE_DATA` | `$BOYUE_ROOT/data_external` | downloaded raw datasets (BEELINE, Perturb-seq h5ad, ...) |
| `BOYUE_CKPT` | `$BOYUE_ROOT/checkpoints` | trained model weights |

Main pipeline stages:

1. **Synthetic SCM cache** — `scripts/data_gen/generate_pa_g200.py` (P+A) and
   `scripts/data_gen/generate_pdn_g200.py` (P+D+N).
2. **Training** — `scripts/train/train_gt_g200_edge_v3.py` (edge model,
   4 seeds) and `scripts/train/train_gt_g200_dir_specialist.py`
   (direction specialists NT `tf_non_tf` and TT `tf_tf`).
3. **Benchmark** — `scripts/eval/benchmark_beeline.py`,
   `scripts/eval/full_evaluation_matrix.py`,
   `scripts/eval/precision_recall_f1.py`, ablation
   (`scripts/eval/ablation_ko_k562.py`), and the cross-specialist / synthetic
   verification scripts in `scripts/eval/`.
4. **Case studies** — `scripts/case/eval_{ad,hcc,nafld,periodontitis}_celltype.py`
   and `scripts/case/eval_zhongliang_celltype.py` (PVL) (5 diseases) +
   `scripts/case/run_*_deg_expanded.py` (stage-level DEG-expanded inference) +
   `scripts/case/go_deg_expanded_all.py` (GO enrichment).
5. **Inference on user data** — the `boyue` package API above.

**Model weights** (`checkpoints/main/`, ≈1.6 GB, not bundled):

```
edge_v3_seed{0,1,2,3}.pt                    # edge existence (4-seed ensemble)
dir_specialist_tf_non_tf_seed{0,1,2,3}.pt   # NT direction specialist (TF→non-TF)
dir_specialist_tf_tf_seed0.pt               # TT direction specialist (TF→TF)
checkpoints/ablation/                       # 7 ablation models (Table 2)
```

Download from the companion HuggingFace model repository:
**https://huggingface.co/holaoctopus/boyuegrn**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="holaoctopus/boyuegrn", repo_type="model",
                  local_dir="./boyuegrn_ckpt")
```

Then set `BOYUE_CKPT` to point at `./boyuegrn_ckpt/checkpoints`.

**External datasets** (download separately; see manuscript Methods):

- **BEELINE** — Pratapa et al., *Nat. Methods* 2020 (MuraliGroup/beeline).
- **HCT116 Perturb-seq** — **X-Atlas/Orion** (Huang et al., bioRxiv 2025, Xaira
  Therapeutics; figshare.plus.29190726 / HuggingFace Xaira-Therapeutics/X-Atlas-Orion).
  The perturbation is **CRISPRi transcriptional knockdown** (dual-sgRNA library
  from Replogle et al., *eLife* 2022, e81856); median on-target knockdown ≈75.4%.
- **K562 Perturb-seq** — Replogle et al., *Cell* 2022 (CRISPRi genome-scale
  Perturb-seq; figshare 15131402).

## 4. License

- **Code** (`boyue/`, `scripts/`): Apache License 2.0 — see `LICENSE`.
