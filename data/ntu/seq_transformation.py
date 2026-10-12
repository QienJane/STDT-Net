import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.ntu.get_raw_skes_data import read_skeleton
from data.ntu.get_raw_denoised_data import select_primary_performer
from data.prepare import check_output, write_prepared_split


TRAIN_SUBJECTS_60 = frozenset((1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,
                             25, 27, 28, 31, 34, 35, 38))
TRAIN_SUBJECTS_120 = TRAIN_SUBJECTS_60 | frozenset((45, 46, 47, 49, 50, 52, 53,
    54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91,
    92, 93, 94, 95, 97, 98, 100, 103))
NAME_PATTERN = re.compile(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})')


def split_for_sample(setup, camera, subject, dataset, protocol):
    allowed = {'ntu60': ('CS', 'CV'), 'ntu120': ('CS', 'CE')}
    if dataset not in allowed or protocol not in allowed[dataset]:
        raise ValueError('Use ntu60 CS/CV or ntu120 CS/CE')
    if protocol == 'CS':
        train = subject in (TRAIN_SUBJECTS_60 if dataset == 'ntu60' else TRAIN_SUBJECTS_120)
    elif protocol == 'CV':
        train = camera in (2, 3)
    else:
        train = setup % 2 == 0
    return 'train' if train else 'test'


def scan_samples(raw_dirs, dataset, protocol, ignored_sample_path=None):
    split_for_sample(1, 1, 1, dataset, protocol)
    missing = set()
    if ignored_sample_path is not None:
        text = Path(ignored_sample_path).read_text(encoding='utf-8')
        missing = {match.group(0) for match in NAME_PATTERN.finditer(text)}
        if not missing:
            raise ValueError('Missing-sample list contains no NTU sample IDs')
    splits = {'train': [], 'test': []}
    seen = set()
    for raw_dir in raw_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.is_dir():
            raise FileNotFoundError(raw_dir)
        for path in sorted(raw_dir.rglob('*.skeleton')):
            match = NAME_PATTERN.fullmatch(path.stem)
            if match is None:
                raise ValueError(f'Unexpected NTU skeleton filename: {path.name}')
            setup, camera, subject, replication, action = map(int, match.groups())
            if dataset == 'ntu60' and (setup > 17 or action > 60):
                continue
            if not 1 <= action <= (60 if dataset == 'ntu60' else 120):
                raise ValueError(f'Invalid action label: {path.name}')
            if path.stem in missing:
                continue
            if path.stem in seen:
                raise ValueError(f'Duplicate NTU sample ID: {path.stem}')
            seen.add(path.stem)
            with path.open(encoding='utf-8') as stream:
                length = int(stream.readline())
            if length <= 0:
                raise ValueError(f'Empty sequence: {path}')
            split = split_for_sample(setup, camera, subject, dataset, protocol)
            splits[split].append({'sample_id': path.stem, 'subject_id': subject,
                'split': split, 'mask_source': 'ntu_tracking_state', 'path': path,
                'length': length, 'label': action - 1})
    if not all(splits.values()):
        raise ValueError('Both training and test samples are required; check raw directories and protocol')
    return splits


def read_sample(record):
    return select_primary_performer(read_skeleton(record['path']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--dataset', choices=('ntu60', 'ntu120'), required=True)
    parser.add_argument('--protocol', choices=('CS', 'CV', 'CE'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ignored_sample_path', type=Path,
                        help='Official missing-skeleton list; no samples are guessed or silently excluded')
    args = parser.parse_args()
    splits = scan_samples(args.raw_dir, args.dataset, args.protocol, args.ignored_sample_path)
    for split in splits:
        check_output(args.output / split)
    for split, records in splits.items():
        write_prepared_split(records, args.output / split, 25, read_sample)
    metadata = {'dataset': args.dataset, 'protocol': args.protocol,
        'coordinates': 'raw_camera_space', 'num_person': 1,
        'primary_performer': 'maximum_sum_of_valid_per_joint_temporal_xyz_variances',
        'missing_file': str(args.ignored_sample_path.resolve()) if args.ignored_sample_path else None,
        'missing_list_applied': args.ignored_sample_path is not None,
        'raw_dirs': [str(path.resolve()) for path in args.raw_dir]}
    (args.output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
