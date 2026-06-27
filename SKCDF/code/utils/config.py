import os

_ECCV_ROOT = os.environ.get('ECCV_ROOT', os.path.expanduser('~/Desktop/ECCV'))
_PROJECT_ROOT = os.environ.get(
    'SKCDF_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
)


class Config:
    def __init__(self, task):
        if task == "synapse":
            self.base_dir = os.path.join(_ECCV_ROOT, 'synapse_data')
            self.save_dir = os.path.join(_ECCV_ROOT, 'synapse_data')
            self.splits_dir = os.path.join(_ECCV_ROOT, 'synapse_splits')
            self.data_dir = os.path.join(_ECCV_ROOT, 'synapse_data', 'npy')
            self.log_dir = os.path.join(_PROJECT_ROOT, 'logs')
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 1500
        else:  # amos
            self.base_dir = os.path.join(_ECCV_ROOT, 'AMOS')
            self.save_dir = os.path.join(_ECCV_ROOT, 'AMOS')
            self.splits_dir = os.path.join(_ECCV_ROOT, 'amos_splits')
            self.data_dir = os.path.join(_ECCV_ROOT, 'AMOS')
            self.log_dir = os.path.join(_PROJECT_ROOT, 'logs')
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 1500


