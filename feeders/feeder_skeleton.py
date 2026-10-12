
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from . import tools


class Feeder(Dataset):
    def __init__(self, data_path, window_size, graph, root_joint=0,
                 reference_bone=(0, 20), training=False):
        directory = Path(data_path)
        self.data = np.load(directory / 'data.npy', mmap_mode='r', allow_pickle=False)
        self.mask = np.load(directory / 'mask.npy', mmap_mode='r', allow_pickle=False)
        self.length = np.load(directory / 'length.npy', mmap_mode='r', allow_pickle=False)
        label_path = directory / 'label.npy'
        self.label = (np.load(label_path, mmap_mode='r', allow_pickle=False)
                      if label_path.exists() else np.full(len(self.data), -1, dtype=np.int64))
        with (directory / 'samples.json').open(encoding='utf-8') as stream:
            self.samples = json.load(stream)
        self.num_point, self.bones = tools.graph_bones(graph)
        self.window_size = int(window_size)
        self.root_joint = int(root_joint)
        self.reference_bone = tuple(reference_bone)
        self.training = bool(training)

        n = len(self.data)
        if (self.data.ndim != 5 or self.data.shape[1] != 3
                or self.data.shape[3:] != (self.num_point, 1)):
            raise ValueError('data.npy must have shape [N,3,L,V,1] matching graph')
        if self.data.dtype != np.float32:
            raise ValueError('data.npy must use float32 coordinates')
        if self.mask.shape != (n, *self.data.shape[2:]) or self.mask.dtype != np.bool_:
            raise ValueError('mask.npy must be boolean [N,L,V,1]')
        if self.length.shape != (n,) or not np.issubdtype(self.length.dtype, np.integer):
            raise ValueError('length.npy must be an integer array of shape [N]')
        if np.any(self.length < 1) or np.any(self.length > self.data.shape[2]):
            raise ValueError('length.npy contains a length outside [1,L]')
        if self.label.shape != (n,) or not np.issubdtype(self.label.dtype, np.integer):
            raise ValueError('label.npy must be an integer array of shape [N]')
        if self.training and np.any(self.label < 0):
            raise ValueError('Training requires label.npy with nonnegative labels')
        fields = {'sample_id', 'subject_id', 'split', 'mask_source'}
        if not isinstance(self.samples, list) or len(self.samples) != n or not n:
            raise ValueError('samples.json must contain one record per sample in a nonempty split')
        if any(not isinstance(record, dict) or not fields <= record.keys()
               or any(record[key] is None or str(record[key]).strip() == '' for key in fields)
               for record in self.samples):
            raise ValueError('Every sample requires sample_id, subject_id, split and mask_source')
        self.sample_name = [str(record['sample_id']) for record in self.samples]
        self.subject_ids = [str(record['subject_id']) for record in self.samples]
        if len(set(self.sample_name)) != n:
            raise ValueError('sample_id values must be unique within a prepared split')
        splits = {str(record['split']) for record in self.samples}
        sources = {str(record['mask_source']) for record in self.samples}
        if len(splits) != 1 or len(sources) != 1:
            raise ValueError('Each prepared directory must have one split and one mask_source')
        self.split = splits.pop()
        self.mask_source = sources.pop()
        if self.mask_source == 'coordinate_proxy':
            raise ValueError('Paper inputs require provided original masks, not coordinate_proxy')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x, mask = tools.prepare_sequence(
            self.data[index], self.mask[index], self.length[index], self.root_joint,
            self.reference_bone, self.window_size, self.training)
        quality = tools.raw_quality(x, mask, self.bones, self.root_joint)
        return x, quality, int(self.label[index]), self.sample_name[index]
