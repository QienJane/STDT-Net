import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.prepare import check_output, write_prepared_split


def read_sample(record):
    with np.load(record['path'], allow_pickle=False) as clip:
        coords = np.asarray(clip['data'], dtype=np.float32)
        mask = np.asarray(clip['mask'])
    if coords.ndim != 3 or coords.shape[1:] != (21, 3) or mask.shape != coords.shape[:2]:
        raise ValueError(f"{record['sample_id']}: require data[T,21,3] and mask[T,21]")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{record['sample_id']}: mask must contain only 0/1 or booleans")
    return coords, mask.astype(bool)


def scan_manifest(manifest):
    manifest = Path(manifest).resolve()
    splits = {'train': [], 'val': [], 'test': []}
    subjects = {}
    seen = set()
    seen_paths = set()
    required = {'sample_id', 'data_path', 'label', 'subject_id', 'split'}
    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        rows = csv.DictReader(stream)
        if not required.issubset(rows.fieldnames or []):
            raise ValueError('CSV requires sample_id,data_path,label,subject_id,split columns')
        for row in rows:
            sample_id, subject = row['sample_id'].strip(), row['subject_id'].strip()
            split, label = row['split'].strip(), int(row['label'])
            path = (manifest.parent / row['data_path']).resolve()
            if not sample_id or not subject or split not in splits or not 0 <= label < 8:
                raise ValueError(f'Invalid sample ID, subject, split, or zero-based 0..7 label: {row}')
            if sample_id in seen or path in seen_paths:
                raise ValueError(f'Duplicate sample ID or source clip: {sample_id}')
            if subject in subjects and subjects[subject] != split:
                raise ValueError(f'Subject {subject} appears in both {subjects[subject]} and {split}')
            subjects[subject] = split
            seen.add(sample_id)
            seen_paths.add(path)
            record = {'sample_id': sample_id, 'subject_id': subject, 'split': split,
                      'label': label, 'path': path, 'mask_source': 'provided_validity'}
            coords, mask = read_sample(record)
            if len(coords) == 0:
                raise ValueError(f'Empty clip: {sample_id}')
            record['length'] = len(coords)
            splits[split].append(record)
    if not all(splits.values()):
        raise ValueError('Manifest must provide nonempty train, val and test subject splits')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    splits = scan_manifest(args.manifest)
    for split in splits:
        check_output(args.output / split)
    for split, records in splits.items():
        write_prepared_split(records, args.output / split, 21, read_sample)
    metadata = {'dataset': 'robogesture8', 'coordinates': 'raw_recorded_space',
                'split_source': str(args.manifest.resolve()), 'subject_disjoint': True,
                'mask_source': 'provided_validity', 'num_person': 1}
    (args.output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
