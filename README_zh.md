# BoYueGRN：基于 scRNA-seq 的有向基因调控网络推断

论文配套开源代码与数据：
**"BoYueGRN: Zero-shot causal discovery of directed gene regulatory networks
from single-cell transcriptomes via amortized inference over synthetic
structural causal models"**

> 英文版见 [README.md](README.md)。

BoYueGRN 是一个摊销式（amortized）因果发现框架，完全基于合成结构因果模型
（SCM）训练。对任意新的单细胞数据集，一次前向传播即可同时推断调控边的
存在概率与调控方向；以 TF 为中心的滑动窗口与非对称融合策略将固定规模的
模型扩展到全转录组覆盖。

**名字的来意。** BoYue（博约）取自苏轼《稼说送张琥》中的名句
"博观而约取，厚积而薄发"。这正是本框架摊销式（amortized）范式的写照：
模型在训练阶段广泛地"博观"海量合成 SCM，把因果结构发现的成本一次性
摊销掉；推断阶段则"约取"，对任意新数据集只需一次前向传播即可完成
全转录组的边与方向推断。

本仓库包含**模型代码与完整的训练 / 基准测试 / 案例研究流程**，全部使用
仓库相对路径（不含机器特定的绝对路径）。模型权重与大型外部数据集
（BEELINE / Perturb-seq h5ad / SCM 缓存 / DEG-expanded pkl）**不随包附带**，
如何指向这些数据见 §3。

***

## 1. 仓库结构

```
BoYueGRN/
├── README.md                       # 英文说明
├── README_zh.md                    # 本文件（中文说明）
├── LICENSE                         # Apache License 2.0
├── requirements.txt                # 基础 Python 依赖
├── requirements-pipeline.txt       # 完整流程依赖（torch、sklearn 等）
├── pyproject.toml                  # pip 可安装包（boyue）
├── boyue/                          # BoYueGRN 模型/推断包
│   ├── model.py                    #   GraphTransformerEncoderV3 + edge/dir 头
│   ├── stats.py                    #   compute_P (Ledoit-Wolf) / compute_D / compute_N
│   ├── data.py                     #   PDNDataset（训练数据加载器）
│   ├── infer.py                    #   load_ensemble / predict_from_expression / directed_edges
│   ├── genome_wide.py              #   全基因组滑动窗口流程
│   ├── disease.py                  #   疾病案例推断 + DEG 计算
│   ├── enrich.py                   #   疾病 GRN 输出的 GO 富集
│   ├── cli.py                      #   `boyue-disease` / `boyue-enrich` 命令行入口
│   └── __init__.py                 #   公共 API + __version__ = "1.0.0"
└── scripts/                        # 完整流程（不含绘图脚本）
    ├── train/                      # 模型训练（edge v3、方向专家模型）
    ├── eval/                       # 基准测试 / cross-specialist 评估
    ├── case/                       # 5 个疾病案例研究推断 + GO
    ├── data_gen/                   # 合成 SCM 缓存生成器（P/A、P/D/N）
    └── _config.py                  # 可用环境变量覆盖的路径配置
```

## 2. boyue 包（模型代码）

```bash
pip install -e .                     # 或：pip install -r requirements-pipeline.txt

# 在随机 200x200 表达矩阵上做快速自检（CPU）
python -c "from boyue import load_ensemble; import numpy as np, torch; \
inf = load_ensemble(edge_ckpts=['checkpoints/main/edge_v3_seed0.pt'], \
dir_ckpts=['checkpoints/main/dir_specialist_tf_non_tf_seed0.pt'], device=torch.device('cpu')); \
print(inf.predict_from_expression(np.random.gamma(2.0, 1.0, (200, 200)).astype(np.float32)).keys())"
```

公共 API：

- `load_ensemble` — 加载训练好的 checkpoint，构造推断对象。
- `predict_from_expression` — 从表达矩阵推断边概率与方向。
- `directed_edges` — 以 DataFrame 形式返回有向边。
- `GenomeWideInferencer` — 全基因组滑动窗口推断。
- `run_disease` / `compute_degs` / `expanded_deg_grn` — 案例研究流程。
- `run_go` — 对案例研究 GRN 输出做 GO 富集。

## 3. 完整流程（训练 / 评估 / 推断）

所有流程代码使用仓库相对路径；外部路径均可通过环境变量覆盖
（定义于 `scripts/_config.py`）：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `BOYUE_ROOT` | 仓库根目录（自动检测） | 包含 `data_external/`、`checkpoints/` 的工作区根目录 |
| `BOYUE_DATA` | `$BOYUE_ROOT/data_external` | 下载的原始数据集（BEELINE、Perturb-seq h5ad 等） |
| `BOYUE_CKPT` | `$BOYUE_ROOT/checkpoints` | 训练好的模型权重 |

主要流程阶段：

1. **合成 SCM 缓存** — `scripts/data_gen/generate_pa_g200.py`（P+A）与
   `scripts/data_gen/generate_pdn_g200.py`（P+D+N）。
2. **训练** — `scripts/train/train_gt_g200_edge_v3.py`（edge 模型，
   4 个随机种子）与 `scripts/train/train_gt_g200_dir_specialist.py`
   （方向专家模型：NT `tf_non_tf` 与 TT `tf_tf`）。
3. **基准测试** — `scripts/eval/benchmark_beeline.py`、
   `scripts/eval/full_evaluation_matrix.py`、
   `scripts/eval/precision_recall_f1.py`、消融实验
   （`scripts/eval/ablation_ko_k562.py`），以及 `scripts/eval/` 下的
   cross-specialist / 合成数据验证脚本。
4. **案例研究** — `scripts/case/eval_{ad,hcc,nafld,periodontitis}_celltype.py`
   与 `scripts/case/eval_zhongliang_celltype.py`（PVL）（共 5 个疾病）+
   `scripts/case/run_*_deg_expanded.py`（分期 DEG-expanded 推断）+
   `scripts/case/go_deg_expanded_all.py`（GO 富集）。
5. **在用户自有数据上推断** — 使用上述 `boyue` 包 API。

**模型权重**（`checkpoints/main/`，约 1.6 GB，不随包附带）：

```
edge_v3_seed{0,1,2,3}.pt                    # 边存在性（4 种子集成）
dir_specialist_tf_non_tf_seed{0,1,2,3}.pt   # NT 方向专家（TF→非TF）
dir_specialist_tf_tf_seed0.pt               # TT 方向专家（TF→TF）
checkpoints/ablation/                       # 7 个消融模型（Table 2）
```

请从配套的 HuggingFace 模型仓库下载：
**https://huggingface.co/holaoctopus/boyuegrn**

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="holaoctopus/boyuegrn", repo_type="model",
                  local_dir="./boyuegrn_ckpt")
```

然后将 `BOYUE_CKPT` 指向 `./boyuegrn_ckpt/checkpoints`。

**外部数据集**（需另行下载；详见论文 Methods）：

- **BEELINE** — Pratapa et al., *Nat. Methods* 2020（MuraliGroup/beeline）。
- **HCT116 Perturb-seq** — **X-Atlas/Orion**（Huang et al., bioRxiv 2025，
  Xaira Therapeutics；figshare.plus.29190726 / HuggingFace
  Xaira-Therapeutics/X-Atlas-Orion）。扰动方式为 **CRISPRi 转录敲低**
  （dual-sgRNA 文库来自 Replogle et al., *eLife* 2022, e81856）；
  on-target 敲低效率中位数约 75.4%。
- **K562 Perturb-seq** — Replogle et al., *Cell* 2022（CRISPRi 全基因组
  Perturb-seq；figshare 15131402）。

## 4. 许可

- **代码**（`boyue/`、`scripts/`）：Apache License 2.0 — 见 `LICENSE`。
