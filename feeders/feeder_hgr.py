
import json

import numpy as np
from torch.utils.data import Dataset

from . import tools


class FeederHGR(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train',
                 random_choose=False, random_shift=False, random_move=False,
                 random_rot=False, window_size=48, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False,
                 graph='graph.hgr_hand.Graph', root_joint=0,
                 reference_bone=(0, 9), training=None, layout='NCTVM'):
        with np.load(data_path, allow_pickle=False) as archive:
            self.data = archive['data']
            if 'label' in archive:
                self.label = archive['label']
            elif label_path:
                with open(label_path, encoding='utf-8') as stream:
                    self.label = np.asarray(json.load(stream))
            else:
                raise ValueError('Legacy HGR training/evaluation requires labels')
        if layout == 'NCVTM':
            self.data = self.data.transpose(0, 1, 3, 2, 4)
        elif layout != 'NCTVM':
            raise ValueError("layout must explicitly be 'NCTVM' or 'NCVTM'")
        self.num_point, self.bones = tools.graph_bones(graph)
        if (self.data.ndim != 5 or self.data.shape[1] != 3
                or self.data.shape[3:] != (self.num_point, 1)):
            raise ValueError('Legacy HGR data must match [N,3,T,V,1] and graph')
        if self.label.shape != (len(self.data),) or not np.issubdtype(self.label.dtype, np.integer):
            raise ValueError('Legacy HGR labels must be an integer array of shape [N]')
        if debug:
            self.data, self.label = self.data[:100], self.label[:100]
        self.window_size = int(window_size)
        self.root_joint = int(root_joint)
        self.reference_bone = tuple(reference_bone)
        self.training = split == 'train' if training is None else bool(training)
        self.bone, self.vel = bone, vel
        self.mask_source = 'coordinate_proxy'
        self.sample_name = [f'{split}_sample_{i:06d}' for i in range(len(self.data))]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return tools.legacy_item(
            self.data[index], self.label[index], self.sample_name[index],
            self.window_size, self.bones, self.root_joint, self.reference_bone,
            self.training, self.bone, self.vel)

    def top_k(self, score, top_k):
        return np.mean(np.any(score.argsort(axis=1)[:, -top_k:] == self.label[:, None], axis=1))
