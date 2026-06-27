#!/usr/bin/env python3
"""Official paper-style Synapse test eval: Dice + ASD on repo test split."""
import argparse
import os
import sys

import numpy as np
import torch
from medpy import metric
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SKCDF', 'code'))
sys.path.insert(0, ROOT)

from models.SKCDF import VNet_Decouple_Attention_ABC  # noqa: E402
from utils import read_data, test_single_case  # noqa: E402
from utils.config import Config  # noqa: E402

OFFICIAL_SPLITS = os.path.join(ROOT, 'data', 'synapse_splits')

PAPER_CLASS_NAMES = [
    'Spleen', 'RK', 'LK', 'Gallbladder', 'Esophagus', 'Liver', 'Stomach',
    'Aorta', 'IVC', 'PSV', 'Pancreas', 'RAG', 'LAG',
]


def read_official_list(split):
    path = os.path.join(OFFICIAL_SPLITS, f'{split}.txt')
    with open(path) as f:
        return sorted(line.strip() for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='synapse_20p/skcdf/fold1')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--speed', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = Config('synapse')
    stride_dict = {0: (32, 16), 1: (64, 16), 2: (128, 32)}
    stride_xy, stride_z = stride_dict[args.speed]

    ckpt = os.path.join(config.log_dir, args.exp.lstrip('/'), 'ckpts/best_model.pth')
    out_dir = os.path.join(config.log_dir, args.exp.lstrip('/'))
    out_path = os.path.join(out_dir, 'evaluation_official_test.txt')
    pred_dir = os.path.join(out_dir, 'predictions_official')
    os.makedirs(pred_dir, exist_ok=True)

    model = VNet_Decouple_Attention_ABC(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False,
    ).cuda()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    print('loaded', ckpt)
    print('official test split:', read_official_list('test'))

    ids_list = read_official_list('test')
    test_cls = list(range(1, config.num_cls))
    values = np.zeros((len(ids_list), len(test_cls), 2))

    import SimpleITK as sitk

    for idx, data_id in enumerate(tqdm(ids_list, desc='official test')):
        image, label = read_data(data_id, task='synapse', normalize=True)
        pred, _ = test_single_case(
            model, image, stride_xy, stride_z, config.patch_size, config.num_cls
        )
        label = label.astype(np.int8)

        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, os.path.join(pred_dir, f'{data_id}.nii.gz'))

        for i in test_cls:
            pred_i = (pred == i)
            label_i = (label == i)
            if pred_i.sum() > 0 and label_i.sum() > 0:
                dice = metric.binary.dc(pred == i, label == i) * 100
                asd = metric.binary.asd(pred == i, label == i)
                values[idx][i - 1] = np.array([dice, asd])
            elif pred_i.sum() > 0 and label_i.sum() == 0:
                dice, asd = 0, 128
            elif pred_i.sum() == 0 and label_i.sum() > 0:
                dice, asd = 0, 128
            else:
                dice, asd = 1, 0
            values[idx][i - 1] = np.array([dice, asd])

    cls_mean = values.mean(axis=0)
    with open(out_path, 'w', encoding='utf-8') as fw:
        fw.write('Official repo test split: ' + ', '.join(ids_list) + '\n')
        fw.write('Checkpoint: ' + ckpt + '\n\n')
        header = f"{'Class':<12} {'Dice':>7} {'ASD':>7}"
        fw.write(header + '\n')
        fw.write('-' * 28 + '\n')
        print(header)
        print('-' * 28)
        for ci, cname in enumerate(PAPER_CLASS_NAMES):
            row = f'{cname:<12} {cls_mean[ci, 0]:>7.1f} {cls_mean[ci, 1]:>7.1f}'
            print(row)
            fw.write(row + '\n')
        avg_dice = cls_mean[:, 0].mean()
        avg_asd = cls_mean[:, 1].mean()
        summary = f'Average Dice: {avg_dice:.2f}\nAverage ASD:  {avg_asd:.2f}\n'
        print(summary.strip())
        fw.write('-' * 28 + '\n')
        fw.write(summary)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
