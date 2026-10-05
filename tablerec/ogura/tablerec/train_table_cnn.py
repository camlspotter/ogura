"""Train a table boundary U-Net from JSON recipes without image files."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .table_cnn import (CHANNELS, SizeBatchSampler, TableTrainingDataset, TableUNet,
                        boundary_loss, boundary_metrics, collate_tables, load_model, text_false_positive_loss)


def run_epoch(model, loader, device, *, optimizer=None, max_batches=None, text_penalty_weight=0):
    model.train(optimizer is not None)
    totals, count = {}, 0
    with torch.set_grad_enabled(optimizer is not None):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            x, target, valid = (batch[k].to(device) for k in ('image', 'target', 'valid'))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = boundary_loss(logits, target, valid)
            if text_penalty_weight:
                penalty = text_false_positive_loss(logits,batch['text_mask'].to(device),valid)
                loss = loss + text_penalty_weight*penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            metrics = boundary_metrics(logits.detach(), target, valid)
            metrics['loss'] = loss.item()
            if text_penalty_weight: metrics['text_false_positive_loss'] = penalty.item()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0)+value*x.shape[0]
            count += x.shape[0]
    if count == 0:
        raise ValueError('Epoch contains no samples')
    return {key: value/count for key, value in totals.items()} | {'samples': count}


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for record in dataset.source.records[:len(dataset)]:
        digest.update((dataset.source.root/record['recipe']).read_bytes())
    return digest.hexdigest()


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def initialize_model(base_channels, checkpoint, device):
    if checkpoint is None:
        return TableUNet(32 if base_channels is None else base_channels).to(device)
    model = load_model(checkpoint, device)
    if base_channels is not None and base_channels != model.base_channels:
        raise ValueError('--base-channels differs from the initial checkpoint')
    return model


def train(args):
    for key in ('epochs', 'batch_size', 'threads'):
        if getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('max_train_batches', 'max_validation_batches'):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f'{key} must be positive')
    if args.lr <= 0 or args.workers < 0:
        raise ValueError('lr must be positive and workers nonnegative')
    text_weight = getattr(args,'text_penalty_weight',0)
    if not np.isfinite(text_weight) or text_weight < 0:
        raise ValueError('text_penalty_weight must be finite and nonnegative')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    training = TableTrainingDataset(args.train, font_dir=args.font_dir, limit=args.train_limit, text_penalty=text_weight>0, max_side=getattr(args,"max_side",1024))
    validation = TableTrainingDataset(args.validation, font_dir=args.font_dir, limit=args.validation_limit, text_penalty=text_weight>0, max_side=getattr(args,"max_side",1024))
    if training.seeds & validation.seeds:
        raise ValueError('Training and validation contain overlapping recipe seeds')
    sampler = SizeBatchSampler(training.sizes, args.batch_size, seed=args.seed)
    validation_sampler = SizeBatchSampler(validation.sizes, args.batch_size, shuffle=False)
    loader_options = dict(collate_fn=collate_tables, num_workers=args.workers)
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=1)
    train_loader = DataLoader(training, batch_sampler=sampler, **loader_options)
    validation_loader = DataLoader(validation, batch_sampler=validation_sampler, **loader_options)
    initial = getattr(args, 'init_checkpoint', None)
    model = initialize_model(args.base_channels, initial, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(max_side=getattr(args,"max_side",1024),base_channels=model.base_channels,
                  init_checkpoint=str(initial.resolve()) if initial is not None else None,
                  init_checkpoint_sha256=hashlib.sha256(initial.read_bytes()).hexdigest() if initial is not None else None,
                  device=device, parameters=sum(p.numel() for p in model.parameters()),
                  channels=list(CHANNELS), train_count=len(training), validation_count=len(validation),
                  train_sha256=dataset_digest(training), validation_sha256=dataset_digest(validation),
                  torch_version=str(torch.__version__))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps(config), flush=True)
    best = float('inf')
    with (args.output/'metrics.jsonl').open('w') as log:
        for epoch in range(1, args.epochs+1):
            sampler.epoch = epoch-1
            train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, max_batches=args.max_train_batches, text_penalty_weight=text_weight)
            validation_metrics = run_epoch(model, validation_loader, device, max_batches=args.max_validation_batches, text_penalty_weight=text_weight)
            record = dict(epoch=epoch, train=train_metrics, validation=validation_metrics)
            log.write(json.dumps(record)+'\n')
            log.flush()
            checkpoint = dict(format_version=1, channels=list(CHANNELS), base_channels=model.base_channels,
                              epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                              metrics=record)
            save_checkpoint(args.output/'last.pt', checkpoint)
            if validation_metrics['loss'] < best:
                best = validation_metrics['loss']
                save_checkpoint(args.output/'best.pt', checkpoint)
            print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-side',type=int,default=1024,help='Shrink-only longest input side limit (default: 1024)')
    parser.add_argument('--font-dir', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--base-channels', type=int, help='Default: checkpoint architecture, or 32 for a new model')
    parser.add_argument('--init-checkpoint', type=Path, help='Initialize weights only; optimizer and epoch count start fresh')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--text-penalty-weight', type=float, default=0, help='Extra zero-target BCE on text regions away from boundaries')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--validation-limit', type=int)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-validation-batches', type=int)
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
