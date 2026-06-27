# SKCDF 复现实验记录（ECCV / DHC Split, seed=0）

> 论文：*A Semantic Knowledge Complementarity based Decoupling Framework for Semi-supervised Class-imbalanced Medical Image Segmentation* (CVPR 2025)  
> 复现环境：XJTLU HPC (`zimuzhang2302@xeclogin.hpc.xjtlu.edu.cn`)，RTX 4090，`conda env: dhc`  
> 数据划分：与 ECCV / DHC 项目一致（`seed=0`），**非** SKCDF 官方 repo 自带 split

---

## 1. 实验总览

| 任务 | Labeled | Unlabeled | Val/Eval | Test | Seed | Max Epoch | Best Val Dice | 训练时长 |
|------|---------|-----------|----------|------|------|-----------|---------------|----------|
| Synapse 20% | 4 vol | 16 vol | 4 vol | 6 vol | 0 | 1500 | **55.34%** (ep.1383) | ~1h51m |
| AMOS 5% | 10 vol | 206 vol | 24 vol | 120 vol | 0 | 1500 | **32.01%** (ep.522) | ~1d19h |

### 超参数

| 参数 | Synapse 20% | AMOS 5% |
|------|-------------|---------|
| `base_lr` | 0.3 | 0.1 |
| `batch_size` | 2 | 2 |
| `num_workers` | 0 | 0 |
| `cps_loss` / `sup_loss` | w_ce+dice | w_ce+dice |
| `cps_w` | 10 | 10 |
| `ema_w` | 0.99 | 0.99 |
| `cps_rampup` | True | True |

### 最佳权重路径

| 任务 | 本地路径 | HPC 路径 |
|------|----------|----------|
| Synapse 20% | `checkpoints/synapse_20p/fold1/best_model.pth` | `/gpfs/work/aac/zimuzhang2302/SKCDF/logs/synapse_20p/skcdf/fold1/ckpts/best_model.pth` |
| AMOS 5% | `checkpoints/amos_5p/fold1/best_model.pth` | `/gpfs/work/aac/zimuzhang2302/SKCDF/logs/amos_5p/skcdf/fold1/ckpts/best_model.pth` |

---

## 2. 数据划分

划分文件见 `splits/synapse/` 与 `splits/amos/`。

### 2.1 Synapse 20%（seed=0）

| Split | 数量 | Volume IDs |
|-------|------|------------|
| **Labeled** | 4 | `0007, 0009, 0021, 0028` |
| **Unlabeled** | 16 | `0002, 0003, 0005, 0006, 0022, 0024, 0025, 0027, 0031, 0033, 0034, 0035, 0037, 0038, 0039, 0040` |
| **Val (eval)** | 4 | `0008, 0010, 0029, 0030` |
| **Test** | 6 | `0001, 0004, 0023, 0026, 0032, 0036` |

说明：训练时 `-se eval` 读取 `val.txt`（代码自动 alias）。Test 集仅用于最终评估。

### 2.2 AMOS 5%（seed=0）

| Split | 数量 | 说明 |
|-------|------|------|
| **Labeled** | 10 | `amos_0078, amos_0158, amos_0170, amos_0192, amos_0391, amos_0396, amos_0410, amos_0514, amos_0548, amos_0578` |
| **Unlabeled** | 206 | 见 `splits/amos/unlabeled_5p.txt` |
| **Eval** | 24 | 见 `splits/amos/eval.txt` |
| **Test** | 120 | 见 `splits/amos/test.txt` |

---

## 3. Test 集四指标结果（DSC / HD95 / NSD / ASD）

评估脚本：`scripts/eval_synapse_4metrics.py`、`scripts/eval_amos_4metrics.py`  
NSD 阈值 τ=2.0（与 ECCV / DHC 惯例一致）。  
原始输出：`results/synapse_test_4metrics.txt`、`results/amos_test_4metrics.txt`

### 3.1 汇总

| 数据集 | Test Cases | **DSC ↑** | **HD95 ↓** | **NSD ↑** | **ASD ↓** |
|--------|------------|-----------|------------|-----------|-----------|
| **Synapse 20%** | 6 | **52.39** | **30.55** | **64.64** | **24.58** |
| **AMOS 5%** | 120 | **33.62** | **67.95** | **36.05** | **53.90** |

### 3.2 Synapse — 逐类

| Class | DSC | HD95 | NSD | ASD |
|-------|-----|------|-----|-----|
| Aorta | 83.0 | 4.6 | 88.6 | 1.1 |
| Gallbladder | 58.0 | 27.7 | 68.7 | 21.9 |
| Spleen | 70.7 | 22.8 | 81.9 | 21.9 |
| Left Kidney | 42.5 | 34.4 | 59.5 | 26.6 |
| Right Kidney | 47.4 | 24.1 | 70.4 | 21.8 |
| Liver | 90.8 | 3.9 | 94.2 | 0.8 |
| Stomach | 59.8 | 14.0 | 68.1 | 4.3 |
| Pancreas | 70.0 | 23.5 | 78.1 | 21.9 |
| Duodenum | 55.6 | 26.1 | 68.0 | 22.3 |
| Portal Vein | 36.4 | 43.9 | 54.5 | 21.7 |
| Vena Cava | 33.1 | 18.6 | 49.4 | 4.7 |
| Left Adrenal | 0.0 | 128.0 | 0.0 | 128.0 |
| Right Adrenal | 33.9 | 25.4 | 59.0 | 22.6 |
| **Mean** | **52.39** | **30.55** | **64.64** | **24.58** |

### 3.3 AMOS — 逐类

| Class | DSC | HD95 | NSD | ASD |
|-------|-----|------|-----|-----|
| Spleen | 64.8 | 20.0 | 67.6 | 6.3 |
| Right Kidney | 1.0 | 86.2 | 2.4 | 73.7 |
| Left Kidney | 52.5 | 72.2 | 56.6 | 32.4 |
| Gallbladder | 6.7 | 119.5 | 6.7 | 119.5 |
| Esophagus | 0.0 | 128.0 | 0.0 | 128.0 |
| Liver | 83.5 | 22.9 | 79.7 | 6.0 |
| Stomach | 50.7 | 35.2 | 50.2 | 10.8 |
| Aorta | 61.1 | 25.2 | 67.6 | 5.6 |
| IVC | 59.1 | 17.8 | 66.5 | 6.0 |
| Pancreas | 37.6 | 27.0 | 43.2 | 8.8 |
| Right Adrenal | 0.0 | 128.0 | 0.0 | 128.0 |
| Left Adrenal | 0.0 | 128.0 | 0.0 | 128.0 |
| Duodenum | 18.2 | 34.9 | 27.6 | 14.8 |
| Bladder | 49.8 | 70.9 | 53.6 | 37.1 |
| Prostate/Uterus | 19.2 | 103.5 | 19.2 | 103.5 |
| **Mean** | **33.62** | **67.95** | **36.05** | **53.90** |

---

## 4. HPC Job 记录

| Job ID | 任务 | 状态 | 耗时 |
|--------|------|------|------|
| 39018 | Smoke (Syn+AMOS) | COMPLETED | 3 min |
| 39019 | Synapse 20% 训练 | COMPLETED | 1h50m |
| 39020 | AMOS 5% 训练 | COMPLETED | 1d19h |
| 39118 | Synapse test 4-metrics | COMPLETED | 1.5 min |
| 39598 | AMOS test 4-metrics | COMPLETED | 3h |

---

## 5. 与论文官方结果对比（仅供参考）

| 设置 | Synapse Test Dice | 说明 |
|------|-------------------|------|
| 论文官方 split（3-fold 均值） | 64.27 ± 1.36 | 官方 labeled/test 划分不同 |
| 本复现（ECCV split, seed=0） | **52.39** (DSC) | 与官方 split 不可直接对比 |

官方 test 划分：`0008, 0027, 0029, 0033, 0037, 0039`（见 `SKCDF/code/data/synapse_splits/`）
