# SKCDF 从零复现指南（ECCV / DHC Split）

本仓库为 [SKCDF (CVPR 2025)](https://github.com/...) 在 **ECCV 统一数据划分** 下的复现 baseline，可直接用于 BIBM 对比实验。

---

## 目录结构

```
bibm_SKCDF_baseline/
├── README.md                 # 本说明
├── EXPERIMENTS.md            # 完整实验结果与划分表
├── REPRODUCTION.md           # 本文件
├── SKCDF/code/               # 训练 / 测试代码（已适配 ECCV 路径）
├── Self_Attention/           # SKCDF 依赖
├── splits/                   # 数据划分（seed=0）
├── checkpoints/              # 最佳权重（Git LFS）
├── results/                  # 评估结果与训练日志
├── scripts/                  # 验证 & 4-metrics 评估脚本
└── slurm/xec/                # XEC HPC 提交脚本
```

---

## Step 0：环境

```bash
conda create -n skcdf python=3.10 -y
conda activate skcdf
pip install torch torchvision tensorboardX einops SimpleITK medpy tqdm scipy numpy
```

> HPC 上可直接 `conda activate dhc`（已验证 torch 2.3 + cu121）。

---

## Step 1：准备数据

数据格式为 **volume-level `.npy`**：

| 数据集 | 路径 | 文件命名 |
|--------|------|----------|
| Synapse | `$ECCV_ROOT/synapse_data/npy/` | `{id}_image.npy`, `{id}_label.npy` |
| AMOS | `$ECCV_ROOT/AMOS/` | `{id}_image.npy`, `{id}_label.npy` |

本地示例：

```bash
export ECCV_ROOT=~/Desktop/ECCV
# 或 HPC: export ECCV_ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF/ECCV
```

**划分文件**：将本仓库 `splits/` 复制或软链到 ECCV：

```bash
mkdir -p $ECCV_ROOT
ln -sfn $(pwd)/splits/synapse $ECCV_ROOT/synapse_splits
ln -sfn $(pwd)/splits/amos     $ECCV_ROOT/amos_splits
# 数据 npy 目录按你的实际路径设置
```

HPC 一键链接（`slurm/xec/setup_data.sh`）：

```bash
export SKCDF_ROOT=/path/to/bibm_SKCDF_baseline
bash slurm/xec/setup_data.sh   # 链接 AD_project 数据 + ECCV splits
```

---

## Step 2：验证数据

```bash
export ECCV_ROOT=~/Desktop/ECCV
export SKCDF_ROOT=$(pwd)
export PYTHONPATH=$SKCDF_ROOT/SKCDF/code:$PYTHONPATH

python scripts/verify_skcdf_data.py   # Synapse 20%
python scripts/verify_amos_data.py    # AMOS 5%
```

---

## Step 3：Smoke 测试

```bash
cd SKCDF/code

# Synapse smoke (1 epoch)
python train_skcdf.py --task synapse --exp synapse_20p/smoke_test --seed 0 \
  -sl labeled_20p -su unlabeled_20p -se eval \
  --max_epoch 1 --batch_size 2 --num_workers 0 --base_lr 0.3 --gpu 0

# AMOS smoke (1 epoch)
python train_skcdf.py --task amos --exp amos_5p/smoke_test --seed 0 \
  -sl labeled_5p -su unlabeled_5p -se eval \
  --max_epoch 1 --batch_size 2 --num_workers 0 --base_lr 0.1 --gpu 0
```

确认生成：`logs/synapse_20p/smoke_test/ckpts/best_model.pth` 等。

---

## Step 4：正式训练

```bash
cd SKCDF/code

# Synapse 20% (seed=0, ~2h on RTX 4090)
python train_skcdf.py --task synapse --exp synapse_20p/skcdf/fold1 --seed 0 \
  -sl labeled_20p -su unlabeled_20p -se eval \
  --max_epoch 1500 --batch_size 2 --num_workers 0 --base_lr 0.3 --gpu 0

# AMOS 5% (seed=0, ~2d on RTX 4090)
python train_skcdf.py --task amos --exp amos_5p/skcdf/fold1 --seed 0 \
  -sl labeled_5p -su unlabeled_5p -se eval \
  --max_epoch 1500 --batch_size 2 --num_workers 0 --base_lr 0.1 --gpu 0
```

**重要**：`num_workers=0`，避免 DataLoader 多进程退出码问题。

HPC 提交：

```bash
sbatch slurm/xec/skcdf_smoke.slurm          # smoke
sbatch slurm/xec/skcdf_synapse_20p.slurm    # Synapse 正式
sbatch slurm/xec/skcdf_amos_5p.slurm        # AMOS 正式
```

---

## Step 5：Test 集 4 指标评估

使用本仓库已训练权重（或 Step 4 产出）：

```bash
# 若使用仓库自带 checkpoint，先复制到 logs 目录
mkdir -p logs/synapse_20p/skcdf/fold1/ckpts logs/amos_5p/skcdf/fold1/ckpts
cp checkpoints/synapse_20p/fold1/best_model.pth logs/synapse_20p/skcdf/fold1/ckpts/
cp checkpoints/amos_5p/fold1/best_model.pth   logs/amos_5p/skcdf/fold1/ckpts/

python scripts/eval_synapse_4metrics.py --exp synapse_20p/skcdf/fold1 --gpu 0
python scripts/eval_amos_4metrics.py     --exp amos_5p/skcdf/fold1     --gpu 0
```

结果写入 `logs/*/evaluation_4metrics.txt`，汇总见 `EXPERIMENTS.md`。

---

## Step 6：使用预训练权重（跳过训练）

```bash
git lfs pull   # 下载 checkpoints/*.pth（约 390MB）
bash scripts/setup_checkpoints.sh
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `ZeroDivisionError` at epoch 0 | 已修复：`train_skcdf.py` 中 `max(args.max_epoch, 1)` |
| DataLoader exit code 1 | 使用 `--num_workers 0` |
| `numpy.bool` 报错 (medpy) | 评估脚本已加 NumPy 2.x 兼容补丁 |
| Synapse val 读不到 | 代码自动将 `eval` alias 到 `val.txt` |
| AMOS 路径 | 数据在 `AMOS/` 而非 `npy/` 子目录 |

---

## 关键修改文件（相对原版 SKCDF）

- `SKCDF/code/utils/config.py` — `ECCV_ROOT` / `SKCDF_ROOT` 环境变量
- `SKCDF/code/utils/__init__.py` — split 别名、`case` 前缀处理
- `SKCDF/code/train_skcdf.py` — 日志路径、`max_epoch=0` 修复
- `scripts/eval_*_4metrics.py` — DSC/HD95/NSD/ASD 评估

详细结果见 [EXPERIMENTS.md](EXPERIMENTS.md)。
