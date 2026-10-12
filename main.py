import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 1


def import_class(name):
    module, attribute = name.rsplit('.', 1)
    return getattr(importlib.import_module(module), attribute)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataset(config, split, training=False):
    args = copy.deepcopy(config[f'{split}_feeder_args'])
    args['training'] = training
    return import_class(config['feeder'])(**args)


def make_loader(dataset, config, device, training=False, generator=None):
    return DataLoader(
        dataset, batch_size=config['batch_size' if training else 'test_batch_size'],
        shuffle=training, num_workers=config.get('num_workers', 4),
        pin_memory=device.type == 'cuda', drop_last=False,
        worker_init_fn=seed_worker, generator=generator,
    )


def validate_labels(dataset, num_class):
    labels = np.asarray(dataset.label)
    if not len(labels) or np.any(labels < 0) or np.any(labels >= num_class):
        raise ValueError(f'Training/evaluation labels must be in [0, {num_class - 1}].')


def check_split_separation(first, second, subject_disjoint=False):
    if set(first.sample_name) & set(second.sample_name):
        raise ValueError('The dataset splits contain overlapping sample IDs.')
    if subject_disjoint:
        if set(first.subject_ids) & set(second.subject_ids):
            raise ValueError('The dataset splits contain overlapping subject IDs.')


@torch.no_grad()
def quality_statistics(loader):
    total = torch.zeros(3, dtype=torch.float64)
    square = torch.zeros(3, dtype=torch.float64)
    count = 0
    for _, quality, _, _ in loader:
        quality = quality.double()
        if not torch.isfinite(quality).all():
            raise ValueError('Non-finite training quality descriptor.')
        total += quality.sum(0)
        square += quality.square().sum(0)
        count += len(quality)
    if count == 0:
        raise ValueError('The training split is empty.')
    mean = total / count
    std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()


def make_optimizer(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': config['weight_decay']},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=config['base_lr'], betas=(0.9, 0.999), eps=1e-8,
    )


def learning_rate(epoch, config):
    epochs = config['num_epoch']
    warmup = min(config.get('warm_up_epoch', 5), max(0, epochs - 2))
    if epoch < warmup:
        return config['base_lr'] * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return config['base_lr'] * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)


def train_epoch(model, loader, optimizer, scaler, device, gradient_clip=1.0):
    model.train()
    loss_sum = correct = count = 0
    for x, quality, label, _ in loader:
        x = x.to(device, non_blocking=True)
        quality = quality.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=scaler.is_enabled()):
            logits = model(x, quality)
            loss = nn.functional.cross_entropy(logits, label)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * len(label)
        correct += int((logits.argmax(1) == label).sum())
        count += len(label)
    return {'loss': loss_sum / count, 'accuracy': correct / count}


@torch.inference_mode()
def evaluate(model, loader, device, collect=False, require_labels=True):
    model.eval()
    loss_sum = correct = labelled = count = 0
    ids, all_labels, all_logits = [], [], []
    for x, quality, label, sample_ids in loader:
        logits = model(x.to(device, dtype=torch.float32, non_blocking=True),
                       quality.to(device, dtype=torch.float32, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite inference logits.')
        label = label.to(device, non_blocking=True)
        valid = label >= 0
        if require_labels and not valid.all():
            raise ValueError('Evaluation requires labels for every sample.')
        if valid.any():
            loss_sum += float(nn.functional.cross_entropy(logits[valid], label[valid], reduction='sum'))
            correct += int((logits[valid].argmax(1) == label[valid]).sum())
            labelled += int(valid.sum())
        count += len(label)
        if collect:
            ids.extend(sample_ids)
            all_labels.append(label.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    if not count:
        raise ValueError('The input split is empty.')
    metrics = {'samples': count, 'labelled_samples': labelled,
               'loss': loss_sum / labelled if labelled else None,
               'accuracy': correct / labelled if labelled else None}
    if not collect:
        return metrics
    logits = np.concatenate(all_logits)
    probabilities = torch.from_numpy(logits).softmax(1).numpy()
    if len(set(ids)) != len(ids):
        raise ValueError('Prediction sample IDs must be unique.')
    predictions = {'sample_ids': np.asarray(ids), 'labels': np.concatenate(all_labels),
                   'logits': logits, 'probabilities': probabilities,
                   'predictions': probabilities.argmax(1)}
    return metrics, predictions


def checkpoint_state(model, optimizer, scaler, config, epoch, best_accuracy,
                     generator, mask_source):
    return {
        'version': CHECKPOINT_VERSION, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'config': copy.deepcopy(config), 'epoch': epoch, 'best_accuracy': best_accuracy,
        'mask_source': mask_source,
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'loader': generator.get_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('version') != CHECKPOINT_VERSION:
        raise ValueError('This checkpoint does not use the current STDT-Net architecture.')
    return checkpoint


def save_checkpoint(state, path):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def run_train(config, device, resume=None):
    work_dir = Path(config['work_dir']).resolve()
    if resume:
        if Path(resume).resolve().parent != work_dir:
            raise ValueError('--resume must belong to --work-dir; keep the original run directory.')
        if config['model_selection'] == 'validation' and not (work_dir / 'best.pt').is_file():
            raise FileNotFoundError('Validation resume requires the original best.pt alongside last.pt.')
    elif (work_dir / 'last.pt').exists():
        raise FileExistsError('Use --resume or a new --work-dir for an existing training run.')
    model = import_class(config['model'])(**config['model_args']).to(device)
    train_data = make_dataset(config, 'train', training=True)
    validate_labels(train_data, config['model_args']['num_class'])
    selection = config['model_selection']
    if selection not in {'final', 'validation'}:
        raise ValueError('model_selection must be final or validation.')
    val_data = make_dataset(config, 'val') if selection == 'validation' else None
    if val_data is not None:
        validate_labels(val_data, config['model_args']['num_class'])
        check_split_separation(train_data, val_data, config.get('subject_disjoint', False))
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = make_loader(train_data, config, device, True, generator)
    val_loader = make_loader(val_data, config, device) if val_data is not None else None
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and config.get('use_amp', True))
    start_epoch, best_accuracy = 0, -1.0
    if resume:
        state = load_checkpoint(resume)
        for key in ('model_args', 'feeder', 'train_feeder_args', 'val_feeder_args', 'model_selection'):
            if state['config'].get(key) != config.get(key):
                raise ValueError(f'Resume configuration differs in {key}.')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best_accuracy = state['epoch'] + 1, state['best_accuracy']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        generator.set_state(state['rng']['loader'])
        if device.type == 'cuda' and state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    else:
        statistics_data = make_dataset(config, 'train', training=False)
        mean, std = quality_statistics(make_loader(statistics_data, config, device))
        model.set_quality_stats(mean, std)
        print(f'Quality mean={mean.tolist()}, std={std.tolist()}', flush=True)
        del statistics_data
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    print(f'Training samples={len(train_data)}, mask_source={train_data.mask_source}', flush=True)
    for epoch in range(start_epoch, config['num_epoch']):
        lr = learning_rate(epoch, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        record = {'epoch': epoch + 1, 'lr': lr,
                  'train': train_epoch(model, train_loader, optimizer, scaler, device,
                                       config.get('gradient_clip', 1.0))}
        improved = False
        if val_loader is not None:
            record['validation'] = evaluate(model, val_loader, device)
            improved = record['validation']['accuracy'] > best_accuracy
            if improved:
                best_accuracy = record['validation']['accuracy']
        state = checkpoint_state(model, optimizer, scaler, config, epoch, best_accuracy,
                                 generator, train_data.mask_source)
        save_checkpoint(state, work_dir / 'last.pt')
        if improved:
            save_checkpoint(state, work_dir / 'best.pt')
        with (work_dir / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    chosen = work_dir / ('best.pt' if selection == 'validation' else 'last.pt')
    print(f'Selected checkpoint: {chosen}', flush=True)


def run_inference(config, device, checkpoint_path, phase, split='test', input_path=None,
                  output=None):
    state = load_checkpoint(checkpoint_path)
    saved = state['config']
    if config['model_args'] != saved['model_args']:
        raise ValueError('Model configuration differs from the checkpoint.')
    runtime = copy.deepcopy(saved)
    runtime['num_workers'] = config.get('num_workers', 4)
    runtime['test_batch_size'] = config['test_batch_size']
    if input_path:
        if phase != 'predict':
            raise ValueError('--input is only used with predict.')
        runtime['feeder'] = 'feeders.feeder_skeleton.Feeder'
        source = saved[f'{split}_feeder_args']
        runtime[f'{split}_feeder_args'] = {
            'data_path': str(input_path), 'window_size': saved['model_args']['num_frame'],
            'graph': saved['model_args']['graph'], 'root_joint': source.get('root_joint', 0),
            'reference_bone': source['reference_bone'],
        }
    dataset = make_dataset(runtime, split)
    if phase == 'eval':
        validate_labels(dataset, saved['model_args']['num_class'])
    model = import_class(saved['model'])(**saved['model_args']).to(device)
    model.load_state_dict(state['model'], strict=True)
    metrics, predictions = evaluate(model, make_loader(dataset, runtime, device), device,
                                    collect=True, require_labels=phase == 'eval')
    target = Path(output) if output else Path(checkpoint_path).parent / f'{phase}_{split}.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **predictions)
    print(json.dumps(metrics), flush=True)
    print(f'Predictions: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['train', 'eval', 'predict'], default='train')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--input', help='Prepared skeleton directory for offline prediction')
    parser.add_argument('--output')
    parser.add_argument('--device', help='cpu or cuda:0')
    parser.add_argument('--work-dir')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    for key in ('config', 'checkpoint', 'resume', 'input', 'output', 'work_dir'):
        if getattr(args, key):
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
    os.chdir(ROOT)
    with open(args.config, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    for argument, key in [('epochs', 'num_epoch'), ('batch_size', 'batch_size'),
                          ('num_workers', 'num_workers'), ('seed', 'seed'), ('work_dir', 'work_dir')]:
        value = getattr(args, argument)
        if value is not None:
            config[key] = value
    if args.batch_size is not None:
        config['test_batch_size'] = args.batch_size
    device = torch.device(args.device or config.get('device', 'cuda:0'))
    if device.type == 'cpu':
        torch.set_num_threads(4)
    seed_everything(config.get('seed', 0))
    if args.phase == 'train':
        if config['num_epoch'] < 1:
            parser.error('--epochs must be positive')
        run_train(config, device, args.resume)
    else:
        if not args.checkpoint:
            parser.error('--checkpoint is required for eval/predict')
        run_inference(config, device, args.checkpoint, args.phase, args.split,
                      args.input, args.output)


if __name__ == '__main__':
    main()
