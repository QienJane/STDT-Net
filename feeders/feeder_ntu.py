
import numpy as np
from torch.utils.data import Dataset

from . import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=64, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False,
                 graph='graph.ntu_rgb_d.Graph', root_joint=0,
                 reference_bone=(0, 20), training=None):
        if split not in {'train', 'test'}:
            raise ValueError('Legacy NTU split must be train or test')
        with np.load(data_path, allow_pickle=False) as archive:
            values = archive[f'x_{split}']
            labels = archive[f'y_{split}']
        if values.ndim != 3 or values.shape[-1] != 150:
            raise ValueError('Legacy NTU NPZ requires data shape [N,T,150]')
        n, frames, _ = values.shape
        self.data = values.reshape(n, frames, 2, 25, 3)[:, :, :1].transpose(0, 4, 1, 3, 2)
        if labels.ndim == 2:
            if labels.shape[0] != n or not np.all((labels > 0).sum(axis=1) == 1):
                raise ValueError('Legacy NTU one-hot labels must have one positive entry per sample')
            self.label = labels.argmax(axis=1).astype(np.int64)
        elif labels.shape == (n,) and np.issubdtype(labels.dtype, np.integer):
            self.label = labels
        else:
            raise ValueError('Legacy NTU labels must be one-hot or integer [N]')
        if debug:
            self.data, self.label = self.data[:100], self.label[:100]
        self.num_point, self.bones = tools.graph_bones(graph)
        if self.num_point != 25:
            raise ValueError('Legacy NTU requires a 25-joint graph')
        self.window_size = int(window_size)
        self.root_joint = int(root_joint)
        self.reference_bone = tuple(reference_bone)
        self.training = split == 'train' if training is None else bool(training)
        self.bone, self.vel = bone, vel
        self.mask_source = 'coordinate_proxy'
        self.sample_name = [f'{split}_{i}' for i in range(len(self.data))]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return tools.legacy_item(
            self.data[index], self.label[index], self.sample_name[index],
            self.window_size, self.bones, self.root_joint, self.reference_bone,
            self.training, self.bone, self.vel)

    def top_k(self, score, top_k):
        return np.mean(np.any(score.argsort(axis=1)[:, -top_k:] == self.label[:, None], axis=1))
