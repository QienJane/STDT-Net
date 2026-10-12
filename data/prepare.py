import json
from pathlib import Path

import numpy as np


PREPARED_FILES = ('data.npy', 'mask.npy', 'length.npy', 'label.npy', 'samples.json')


def check_output(output_dir):
    output_dir = Path(output_dir)
    for name in PREPARED_FILES:
        if (output_dir / name).exists():
            raise FileExistsError(f'{output_dir / name} already exists; choose a new output directory')


def write_prepared_split(records, output_dir, num_joints, read_sample):
    if not records:
        raise ValueError(f'No samples for {output_dir}')
    output_dir = Path(output_dir)
    check_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count, max_frames = len(records), max(record['length'] for record in records)
    if max_frames <= 0:
        raise ValueError('Sequences must contain at least one frame')
    make_map = np.lib.format.open_memmap
    data = make_map(output_dir / 'data.npy', mode='w+', dtype=np.float32,
                    shape=(count, 3, max_frames, num_joints, 1))
    mask = make_map(output_dir / 'mask.npy', mode='w+', dtype=bool,
                    shape=(count, max_frames, num_joints, 1))
    lengths = make_map(output_dir / 'length.npy', mode='w+', dtype=np.int64, shape=(count,))
    labels = make_map(output_dir / 'label.npy', mode='w+', dtype=np.int64, shape=(count,))
    samples = []
    for index, record in enumerate(records):
        coords, valid = read_sample(record)
        frames = record['length']
        if coords.shape != (frames, num_joints, 3) or valid.shape != (frames, num_joints):
            raise ValueError(f"{record['sample_id']}: inconsistent coordinates, mask, or length")
        valid = np.asarray(valid, dtype=bool) & np.isfinite(coords).all(axis=-1)
        data[index] = 0
        mask[index] = False
        data[index, :, :frames, :, 0] = np.where(valid[..., None], coords, 0).transpose(2, 0, 1)
        mask[index, :frames, :, 0] = valid
        lengths[index], labels[index] = frames, record['label']
        samples.append({key: record[key] for key in ('sample_id', 'subject_id', 'split', 'mask_source')})
        if (index + 1) % 1000 == 0:
            print(f'{output_dir.name}: {index + 1}/{count}', flush=True)
    for array in (data, mask, lengths, labels):
        array.flush()
    with (output_dir / 'samples.json').open('w', encoding='utf-8') as stream:
        json.dump(samples, stream, ensure_ascii=False, indent=2)
    print(f'{output_dir}: {count} samples, maximum original length {max_frames}', flush=True)
