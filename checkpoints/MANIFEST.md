# Checkpoint Manifest

| 任务 | 文件 | SHA256 | 大小 | Best Val Dice | Best Epoch |
|------|------|--------|------|---------------|------------|
| Synapse 20% | `synapse_20p/fold1/best_model.pth` | `169bdab86c1373a75507a3b870883f08a1c544dff427a1fff28e3d12b0062150` | 195 MB | 55.34% | 1383 |
| AMOS 5% | `amos_5p/fold1/best_model.pth` | `eb31b53420b221b057209853d99c3da4baf2451f38a67881ac37ebd61dac4de0` | 195 MB | 32.01% | 522 |

## 用法

复制到训练日志目录后运行评估：

```bash
bash scripts/setup_checkpoints.sh
python scripts/eval_synapse_4metrics.py --exp synapse_20p/skcdf/fold1 --gpu 0
python scripts/eval_amos_4metrics.py     --exp amos_5p/skcdf/fold1     --gpu 0
```

## 从 HPC 下载（备选）

```bash
HPC_USER=zimuzhang2302
HPC_HOST=xeclogin.hpc.xjtlu.edu.cn
HPC_ROOT=/gpfs/work/aac/zimuzhang2302/SKCDF

scp ${HPC_USER}@${HPC_HOST}:${HPC_ROOT}/logs/synapse_20p/skcdf/fold1/ckpts/best_model.pth \
    checkpoints/synapse_20p/fold1/
scp ${HPC_USER}@${HPC_HOST}:${HPC_ROOT}/logs/amos_5p/skcdf/fold1/ckpts/best_model.pth \
    checkpoints/amos_5p/fold1/
```

> 权重文件通过 **Git LFS** 托管（单文件 ~195MB，超出 GitHub 普通文件 100MB 限制）。
