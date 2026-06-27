# SKCDF Baseline Reproduction (ECCV Split)

SKCDF (CVPR 2025) 在 **ECCV / DHC 统一数据划分**（seed=0）下的复现 baseline，含最佳权重、划分文件、训练脚本与完整 test 评估结果。

## 快速链接

- **[EXPERIMENTS.md](EXPERIMENTS.md)** — 数据划分、四指标结果详表、Job 记录
- **[REPRODUCTION.md](REPRODUCTION.md)** — 从零复现步骤（环境 → 数据 → 训练 → 评估）
- **[checkpoints/MANIFEST.md](checkpoints/MANIFEST.md)** — 最佳权重清单与 SHA256

## 复现结果摘要（Test 集，seed=0）

| 任务 | DSC ↑ | HD95 ↓ | NSD ↑ | ASD ↓ |
|------|-------|--------|-------|-------|
| Synapse 20% | **52.39** | 30.55 | 64.64 | 24.58 |
| AMOS 5% | **33.62** | 67.95 | 36.05 | 53.90 |

## 克隆与获取权重

```bash
git clone https://github.com/linvii77/bibm_SKCDF_baseline.git
cd bibm_SKCDF_baseline
git lfs install
git lfs pull
bash scripts/setup_checkpoints.sh
```

## 评估（使用预训练权重）

```bash
export ECCV_ROOT=/path/to/ECCV   # 含 synapse_data/npy 与 AMOS/
export SKCDF_ROOT=$(pwd)
export PYTHONPATH=$SKCDF_ROOT/SKCDF/code:$PYTHONPATH

python scripts/eval_synapse_4metrics.py --gpu 0
python scripts/eval_amos_4metrics.py --gpu 0
```

## 目录

| 路径 | 说明 |
|------|------|
| `SKCDF/code/` | 训练 / 测试核心代码 |
| `splits/` | Synapse 20% & AMOS 5% 划分 |
| `checkpoints/` | 最佳 `best_model.pth`（Git LFS） |
| `results/` | test 四指标原始输出 + 训练日志 |
| `scripts/` | 验证、评估、checkpoint 安装脚本 |
| `slurm/xec/` | XEC HPC SLURM 脚本 |

## 引用

```bibtex
@inproceedings{zhang2025semantic,
  title={A Semantic Knowledge Complementarity based Decoupling Framework for Semi-supervised Class-imbalanced Medical Image Segmentation},
  author={Zhang, Zheng and Yin, Guanchun and Zhang, Bo and Liu, Wu and Zhou, Xiuzhuang and Wang, Wendong},
  booktitle={CVPR},
  year={2025}
}
```

## 致谢

基于 SKCDF 官方实现，适配 ECCV 数据格式与划分。原始项目见 SKCDF 官方仓库。
