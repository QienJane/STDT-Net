import numpy as np


def motion_variance(joints, mask):
    score = 0.0
    for joint in range(joints.shape[1]):
        observed = joints[mask[:, joint], joint]
        if len(observed) >= 2:
            score += float(observed.astype(np.float64).var(axis=0).sum())
    return score


def select_primary_performer(sequence):
    bodies = sequence['data']
    if not bodies:
        return (np.zeros((sequence['num_frames'], 25, 3), dtype=np.float32),
                np.zeros((sequence['num_frames'], 25), dtype=bool))
    body = max(bodies.values(), key=lambda b: motion_variance(b['joints'], b['mask']))
    return body['joints'], body['mask']


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from get_raw_skes_data import read_skeleton
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('skeleton', type=Path)
    args = parser.parse_args()
    coords, mask = select_primary_performer(read_skeleton(args.skeleton))
    print(f'frames={len(coords)}, valid_joints={int(mask.sum())}, motion={motion_variance(coords, mask):.6g}')
