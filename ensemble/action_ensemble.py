import argparse
import csv
from pathlib import Path
import pickle

import numpy as np


def load_scores(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            ids = data['sample_ids'].astype(str)
            scores = data['scores'] if 'scores' in data else data['probabilities']
            labels = data['labels'] if 'labels' in data else None
    else:
        with path.open('rb') as stream:
            data = pickle.load(stream)
        ids = np.asarray(list(data), dtype=str)
        scores = np.asarray(list(data.values()), dtype=np.float32)
        labels = None
    if scores.ndim != 2 or len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError('Scores must be a matrix with one unique sample ID per row.')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite.')
    return ids, scores, labels


def fuse_scores(streams, weights=None):
    if not streams:
        raise ValueError('At least one score stream is required.')
    weights = np.ones(len(streams)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(streams),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Use one nonnegative weight per stream, with a positive total.')
    weights = weights / weights.sum()
    ids, first_scores, _ = streams[0]
    ids = np.asarray(ids, dtype=str)
    if len(set(ids)) != len(ids):
        raise ValueError('Sample IDs must be unique.')
    result = np.zeros_like(first_scores, dtype=np.float64)
    labels = np.full(len(ids), -1, dtype=np.int64)
    for (stream_ids, scores, stream_labels), weight in zip(streams, weights):
        positions = {str(name): i for i, name in enumerate(stream_ids)}
        if len(positions) != len(stream_ids) or set(positions) != set(ids):
            raise ValueError('Every stream must contain the same unique sample IDs.')
        order = np.asarray([positions[name] for name in ids])
        scores = np.asarray(scores)[order]
        if scores.shape != result.shape or not np.isfinite(scores).all():
            raise ValueError('Score shapes/classes must agree and values must be finite.')
        result += weight * scores
        if stream_labels is not None:
            incoming = np.asarray(stream_labels, dtype=np.int64)[order]
            present = incoming >= 0
            if np.any(present & (labels >= 0) & (incoming != labels)):
                raise ValueError('Labels disagree for the same sample ID.')
            labels[present] = incoming[present]
    return ids, result.astype(np.float32), labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', nargs='+', required=True, help='Prediction NPZs or legacy ID-to-score pickle dictionaries')
    parser.add_argument('--weights', nargs='+', type=float, help='Defaults to equal positive weights')
    parser.add_argument('--labels', help='Optional CSV with sample_id,label (zero-based labels)')
    parser.add_argument('--output', default='ensemble_scores.npz')
    args = parser.parse_args()
    ids, scores, labels = fuse_scores([load_scores(path) for path in args.scores], args.weights)
    if args.labels:
        with open(args.labels, encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        mapping = {row['sample_id']: int(row['label']) for row in rows}
        if len(mapping) != len(rows) or set(mapping) != set(ids):
            raise ValueError('Label CSV must match the unique prediction sample IDs.')
        provided = np.asarray([mapping[name] for name in ids])
        if np.any((labels >= 0) & (labels != provided)):
            raise ValueError('CSV labels disagree with prediction labels.')
        labels = provided
    predictions = scores.argmax(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, sample_ids=ids, scores=scores, labels=labels, predictions=predictions)
    labelled = labels >= 0
    if labelled.any():
        print(f'Accuracy: {(predictions[labelled] == labels[labelled]).mean():.6f}')
    print(f'Samples: {len(ids)}; output: {output}')


if __name__ == '__main__':
    main()
