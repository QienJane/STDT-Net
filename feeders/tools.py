
from importlib import import_module

import numpy as np


def graph_bones(graph):
    module, name = graph.rsplit('.', 1)
    skeleton = getattr(import_module(module), name)()
    edges = sorted({tuple(sorted((int(a), int(b))))
                    for a, b in skeleton.inward if a != b})
    bones = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if bones.size and (bones.min() < 0 or bones.max() >= skeleton.num_node):
        raise ValueError(f'{graph} contains out-of-range bone indices')
    return skeleton.num_node, bones


def coordinate_proxy_mask(data):
    return np.isfinite(data).all(axis=0) & np.any(data != 0, axis=0)


def raw_quality(data, mask, bones, root_joint=0):
    x = np.asarray(data, dtype=np.float32)[..., 0]
    valid = np.asarray(mask, dtype=bool)[..., 0]
    nonroot = np.arange(x.shape[2]) != root_joint
    dropout = 1.0 - valid[:, nonroot].mean()
    pairs = valid[1:, nonroot] & valid[:-1, nonroot]
    displacement = np.linalg.norm(np.diff(x[:, :, nonroot], axis=1), axis=0)
    mean_displacement = displacement[pairs].mean() if pairs.any() else 0.0

    bone_variability = 0.0
    if len(bones):
        a, b = np.asarray(bones, dtype=np.int64).T
        lengths = np.linalg.norm(x[:, :, a] - x[:, :, b], axis=0)
        eligible = valid[:, a] & valid[:, b]
        counts = eligible.sum(axis=0)
        usable = counts >= 2
        if usable.any():
            means = (lengths * eligible).sum(axis=0) / np.maximum(counts, 1)
            variances = (((lengths - means) ** 2) * eligible).sum(axis=0)
            variances /= np.maximum(counts, 1)
            bone_variability = np.sqrt(variances[usable]).mean()
    return np.asarray([dropout, mean_displacement, bone_variability], dtype=np.float32)


def prepare_sequence(data, mask, length, root_joint, reference_bone,
                     window_size, training=False):
    data = np.asarray(data, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if data.ndim != 4 or data.shape[0] != 3 or data.shape[-1] != 1:
        raise ValueError('Expected skeleton shape [3,L,V,1]')
    if mask.shape != data.shape[1:]:
        raise ValueError('Expected mask shape [L,V,1] matching the skeleton')
    if not 0 < int(length) <= data.shape[1] or window_size < 1:
        raise ValueError('length must be within [1,L] and window_size must be positive')
    if not 0 <= root_joint < data.shape[2]:
        raise ValueError('root_joint is outside the skeleton')
    a, b = reference_bone
    if a == b or min(a, b) < 0 or max(a, b) >= data.shape[2]:
        raise ValueError('reference_bone must contain two distinct valid joint indices')

    x = data[:, :int(length)].copy()
    valid = mask[:int(length)].copy() & np.isfinite(x).all(axis=0)
    valid &= valid[:, root_joint:root_joint + 1]
    x = np.where(valid[None], x, 0.0)
    x -= x[:, :, root_joint:root_joint + 1]
    x = np.where(valid[None], x, 0.0)
    reference_valid = valid[:, a, 0] & valid[:, b, 0]
    reference_lengths = np.linalg.norm(x[:, :, a, 0] - x[:, :, b, 0], axis=0)
    scale = reference_lengths[reference_valid].mean() if reference_valid.any() else 0.0
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError('The clip has no positive reference-bone scale from valid endpoints')
    x /= scale

    if training:
        if np.random.random() < 0.5:
            angle = np.deg2rad(np.random.uniform(-15.0, 15.0))
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            x = np.einsum('ij,jtvm->itvm', rotation, x)
        if np.random.random() < 0.5:
            x *= np.float32(np.random.uniform(0.9, 1.1))
        if np.random.random() < 0.5:
            keep = max(1, int(round(x.shape[1] * np.random.uniform(0.9, 1.0))))
            start = np.random.randint(x.shape[1] - keep + 1)
            x, valid = x[:, start:start + keep], valid[start:start + keep]

    positions = np.linspace(0, x.shape[1] - 1, int(window_size))
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, x.shape[1] - 1)
    weight = (positions - left).astype(np.float32)[None, :, None, None]
    x = x[:, left] * (1.0 - weight) + x[:, right] * weight
    valid = valid[np.floor(positions + 0.5).astype(np.int64)]
    x = np.where(valid[None], x, 0.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(x), valid


def apply_modality(data, bones, bone=False, velocity=False, root_joint=0):
    if bone:
        bone_data = np.zeros_like(data)
        visited = {root_joint}
        pending = [root_joint]
        for parent in pending:
            for a, b in bones:
                child = b if a == parent else a if b == parent else None
                if child is not None and child not in visited:
                    bone_data[:, :, child] = data[:, :, child] - data[:, :, parent]
                    visited.add(child)
                    pending.append(child)
        data = bone_data
    if velocity:
        motion = np.zeros_like(data)
        motion[:, :-1] = data[:, 1:] - data[:, :-1]
        data = motion
    return data


def legacy_item(data, label, sample_id, window_size, bones, root_joint,
                reference_bone, training, bone=False, velocity=False, length=None):
    mask = coordinate_proxy_mask(data)
    occupied = np.flatnonzero(mask.any(axis=(1, 2)))
    if not len(occupied):
        raise ValueError(f'{sample_id}: no nonzero finite skeleton coordinates')
    if length is None:
        length = int(occupied[-1]) + 1
    x, mask = prepare_sequence(data, mask, length, root_joint, reference_bone,
                               window_size, training)
    quality = raw_quality(x, mask, bones, root_joint)
    x = apply_modality(x, bones, bone, velocity, root_joint)
    return x, quality, int(label), str(sample_id)
