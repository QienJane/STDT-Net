from contextlib import nullcontext
from pathlib import Path

import numpy as np


def read_skeleton(source):
    context = nullcontext(source) if hasattr(source, 'read') else open(source, encoding='utf-8')
    with context as stream:
        num_frames = int(stream.readline())
        if num_frames <= 0:
            raise ValueError('NTU sequence must contain at least one frame')
        bodies = {}
        for frame in range(num_frames):
            num_bodies = int(stream.readline())
            for _ in range(num_bodies):
                body_id = stream.readline().split()[0]
                num_joints = int(stream.readline())
                if num_joints != 25:
                    raise ValueError(f'NTU body {body_id}: expected 25 joints, got {num_joints}')
                if body_id not in bodies:
                    bodies[body_id] = {
                        'joints': np.zeros((num_frames, 25, 3), dtype=np.float32),
                        'mask': np.zeros((num_frames, 25), dtype=bool),
                        'interval': [],
                    }
                body = bodies[body_id]
                body['interval'].append(frame)
                for joint in range(25):
                    values = stream.readline().split()
                    if len(values) != 12:
                        raise ValueError('NTU joint row must contain 12 values including trackingState')
                    xyz = np.asarray(values[:3], dtype=np.float32)
                    valid = int(values[11]) > 0 and np.isfinite(xyz).all() and np.any(xyz != 0)
                    if valid:
                        body['joints'][frame, joint] = xyz
                        body['mask'][frame, joint] = True
    return {'num_frames': num_frames, 'data': bodies}


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes=None, frames_drop_logger=None):
    result = read_skeleton(Path(skes_path) / f'{ske_name}.skeleton')
    result['name'] = ske_name
    present = np.zeros(result['num_frames'], dtype=bool)
    for body in result['data'].values():
        present[body['interval']] = True
    missing = np.flatnonzero(~present)
    if missing.size and frames_drop_skes is not None:
        frames_drop_skes[ske_name] = missing
    if missing.size and frames_drop_logger is not None:
        frames_drop_logger.info('%s: retained empty frames %s', ske_name, missing.tolist())
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('skeleton', type=Path)
    args = parser.parse_args()
    sequence = read_skeleton(args.skeleton)
    print(f"frames={sequence['num_frames']}, bodies={len(sequence['data'])}")
