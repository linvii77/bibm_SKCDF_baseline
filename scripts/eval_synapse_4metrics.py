#!/usr/bin/env python3
"""Evaluate SKCDF Synapse test set with DSC, HD95, NSD, ASD (ECCV convention)."""
import argparse
import os
import sys

import numpy as np
import numpy as _np_compat

if not hasattr(_np_compat, 'bool'):
    _np_compat.bool = bool
import torch
from medpy import metric
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SKCDF', 'code'))
sys.path.insert(0, ROOT)

from models.SKCDF import VNet_Decouple_Attention_ABC  # noqa: E402
from utils import read_list, read_data, test_single_case  # noqa: E402
from utils.config import Config  # noqa: E402

SYNAPSE_CLASS_NAMES = [
    'Aorta', 'Gallbladder', 'Spleen', 'Left Kidney', 'Right Kidney',
    'Liver', 'Stomach', 'Pancreas', 'Duodenum',
    'Portal Vein', 'Vena Cava', 'Left Adrenal', 'Right Adrenal',
]


def surface_distances(pred, label):
    pred = pred.astype(bool)
    label = label.astype(bool)
    pred_surf = pred ^ binary_erosion(pred, iterations=1)
    label_surf = label ^ binary_erosion(label, iterations=1)
    dt_label = distance_transform_edt(~label)
    dt_pred = distance_transform_edt(~pred)
    d_p2l = dt_label[pred_surf].astype(np.float32)
    d_l2p = dt_pred[label_surf].astype(np.float32)
    return d_p2l, d_l2p


def nsd(pred, label, tau):
    if pred.sum() == 0 and label.sum() == 0:
        return 100.0
    if pred.sum() == 0 or label.sum() == 0:
        return 0.0
    d_p2l, d_l2p = surface_distances(pred, label)
    if len(d_p2l) == 0 and len(d_l2p) == 0:
        return 100.0
    n = np.sum(d_p2l <= tau) + np.sum(d_l2p <= tau)
    d = len(d_p2l) + len(d_l2p)
    return float(n) / float(d) * 100.0


def metrics_one_class(pred_i, label_i, tau):
    p = pred_i.astype(bool)
    l = label_i.astype(bool)
    if not p.any() and not l.any():
        return 100.0, 0.0, 100.0, 0.0
    if not p.any() or not l.any():
        return 0.0, 128.0, 0.0, 128.0
    dc = metric.binary.dc(p, l) * 100.0
    hd = metric.binary.hd95(p, l)
    asd_ = metric.binary.asd(p, l)
    nsd_ = nsd(p, l, tau)
    return float(dc), float(hd), float(nsd_), float(asd_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='synapse_20p/skcdf/fold1')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--task', type=str, default='synapse')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--nsd_tau', type=float, default=2.0)
    parser.add_argument('--speed', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = Config(args.task)
    stride_dict = {0: (32, 16), 1: (64, 16), 2: (128, 32)}
    stride_xy, stride_z = stride_dict[args.speed]

    ckpt = os.path.join(config.log_dir, args.exp.lstrip('/'), 'ckpts/best_model.pth')
    out_dir = os.path.join(config.log_dir, args.exp.lstrip('/'))
    out_path = os.path.join(out_dir, 'evaluation_4metrics.txt')

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

    ids_list = read_list(args.split, task=args.task)
    test_cls = list(range(1, config.num_cls))
    values = np.zeros((len(ids_list), len(test_cls), 4), dtype=np.float64)

    for idx, data_id in enumerate(tqdm(ids_list, desc='test volumes')):
        image, label = read_data(data_id, task=args.task, normalize=True)
        pred, _ = test_single_case(
            model, image, stride_xy, stride_z, config.patch_size, config.num_cls
        )
        label = label.astype(np.int8)
        for ci, cls in enumerate(test_cls):
            values[idx, ci] = metrics_one_class(pred == cls, label == cls, args.nsd_tau)

    cls_mean = values.mean(axis=0)
    with open(out_path, 'w', encoding='utf-8') as fw:
        header = f"{'Class':<20} {'DSC':>7} {'HD95':>7} {'NSD':>7} {'ASD':>7}"
        fw.write(header + '\n')
        fw.write('-' * 50 + '\n')
        print(header)
        print('-' * 50)
        for ci, cname in enumerate(SYNAPSE_CLASS_NAMES):
            row = (
                f'{cname:<20} {cls_mean[ci, 0]:>7.1f} {cls_mean[ci, 1]:>7.1f}'
                f' {cls_mean[ci, 2]:>7.1f} {cls_mean[ci, 3]:>7.1f}'
            )
            print(row)
            fw.write(row + '\n')
        fw.write('-' * 50 + '\n')
        summary = (
            f'Mean  DSC={cls_mean[:, 0].mean():.2f}  HD95={cls_mean[:, 1].mean():.2f}'
            f'  NSD={cls_mean[:, 2].mean():.2f}  ASD={cls_mean[:, 3].mean():.2f}'
        )
        print(summary)
        fw.write(summary + '\n')
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
