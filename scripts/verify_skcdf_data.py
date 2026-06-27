#!/usr/bin/env python3
"""Verify ECCV paths and splits for SKCDF before training."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SKCDF', 'code'))
sys.path.insert(0, ROOT)

from utils import read_list, read_data  # noqa: E402
from utils.config import Config  # noqa: E402


def check_task(task, labeled_split, unlabeled_split):
    print(f'\n=== {task} ===')
    cfg = Config(task)
    print('ECCV_ROOT:', os.environ.get('ECCV_ROOT', '(default)'))
    print('data_dir:', cfg.data_dir)
    print('splits_dir:', cfg.splits_dir)

    for split in [labeled_split, unlabeled_split, 'eval', 'test']:
        ids = read_list(split, task=task)
        print(f'  {split}: {len(ids)} volumes')
        if not ids:
            raise RuntimeError(f'empty split: {split}')
        img, lbl = read_data(ids[0], task=task)
        print(f'    sample {ids[0]}: image {img.shape}, label dtype {lbl.dtype}')
    print(f'  {task} data OK')


if __name__ == '__main__':
    check_task('synapse', 'labeled_20p', 'unlabeled_20p')
    check_task('amos', 'labeled_5p', 'unlabeled_5p')
    print('\nAll checks passed.')
