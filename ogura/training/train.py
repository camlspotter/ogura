"""Train a line CNN with on-the-fly rendering and recoverable checkpoints.

Run: python -m ogura.training.train --help
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import fontTools
import PIL
import numpy as np
from PIL import features
import torch
from torch.utils.data import DataLoader, Dataset

from ogura.text_common import ROOT
from .checkpoint import Checkpoints, restore_rng, rng_state
from .model import LineCNN, ctc_loss
from .metrics import MetricsLog, add_totals, batch_totals, decode, empty_totals, summary
from .render import BatchRenderer, Vocabulary, font_characters, parameters_for_sample, replace_unsupported


@dataclass(frozen=True)
class TrainConfig:
    text: Path = ROOT / 'datasets/final_50_len20_25_hiragana_mix5/train.txt'
    vocabulary: Path = ROOT / 'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl'
    font: Path = ROOT / 'corpus/fonts/NotoSansCJKjp-Regular.otf'
    run_dir: Path = ROOT / 'runs/noto48'
    device: str = 'auto'
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    lr_decay: float = 0.95
    channels: int = 32
    seed: int = 20260915
    save_every: int = 500
    workers: int = 0
    font_size_min: int = 40
    font_size_max: int = 40
    amp: bool = False
    deterministic: bool = False
    resume: bool = False
    max_steps: int | None = None
    limit: int | None = None
    threads: int = 4
    log_every: int = 10
    log_samples: int = 3


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.train.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another trainer is using {directory}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_data(config, vocabulary):
    supported = font_characters(str(config.font.resolve()))
    vocabulary_set = set(vocabulary.characters)
    records = []
    replaced_lines = []
    replaced_occurrences = Counter()
    missing_counts = Counter()
    total = 0
    with config.text.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.removesuffix('\n')
            if not text or any(c in text for c in '\r\n\x85\u2028\u2029'):
                raise ValueError(f'Invalid text at line {line_number}')
            unknown = set(text) - vocabulary_set
            if unknown:
                raise ValueError(f'Unknown labels at line {line_number}: {unknown}')
            total += 1
            missing = {c for c in text if ord(c) not in supported}
            if missing:
                missing_counts.update(missing)
                replaced_lines.append(line_number)
                replaced_occurrences.update(c for c in text if c in missing)
                if ' ' not in vocabulary.ids:
                    raise ValueError('Vocabulary must contain replacement space U+0020')
            sample_id = hashlib.sha256(text.encode()).hexdigest()
            records.append((replace_unsupported(text, str(config.font.resolve())), sample_id))
    if not records:
        raise ValueError('Text dataset is empty')
    report = {
        'total_rows': total, 'unchanged_rows': total - len(replaced_lines), 'excluded_rows': 0,
        'replacement': 'U+0020', 'replaced_rows': len(replaced_lines),
        'replaced_characters': sum(replaced_occurrences.values()),
        'replaced_line_numbers': replaced_lines,
        'unsupported_characters': [
            {'character': c, 'codepoint': f'U+{ord(c):04X}', 'rows': missing_counts[c], 'occurrences': replaced_occurrences[c]}
            for c in vocabulary.characters if ord(c) not in supported
        ],
    }
    if config.limit is not None:
        records = records[:config.limit]
    report['training_rows'] = len(records)
    return records, report


class EpochDataset(Dataset):
    def __init__(self, records, order, config, epoch):
        self.records, self.order, self.config, self.epoch = records, order, config, epoch

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        from .render import Sample
        text, sample_id = self.records[self.order[index]]
        return Sample(text, parameters_for_sample(
            self.config.font.resolve(), self.config.seed, self.epoch, sample_id,
            self.config.font_size_min, self.config.font_size_max,
        ), sample_id)


def identity_for(config, device):
    # Paths and operational settings may differ after copying a run to another machine.
    settings = asdict(config)
    for key in ('text', 'vocabulary', 'font', 'run_dir', 'device', 'resume', 'max_steps',
                'workers', 'save_every', 'log_every', 'log_samples', 'epochs'):
        settings.pop(key)
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code.update(path.name.encode()); code.update(path.read_bytes())
    return {
        'format': 1, 'text_sha256': sha256(config.text),
        'vocabulary_sha256': sha256(config.vocabulary), 'font_sha256': sha256(config.font),
        'training_code_sha256': code.hexdigest(), 'settings': settings,
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'pillow': PIL.__version__, 'fonttools': fontTools.__version__,
                    'freetype': features.version('freetype2'), 'device_type': device.type},
    }


def train(config: TrainConfig, on_step=None):
    """on_step is an optional observer called only after a complete update."""
    positive = (config.batch_size, config.epochs, config.channels, config.save_every,
                config.threads, config.log_every)
    if min(positive) < 1 or config.log_samples < 0 or config.workers < 0 or not config.learning_rate > 0:
        raise ValueError('Invalid training configuration')
    if not 0 < config.lr_decay <= 1 or not 1 <= config.font_size_min <= config.font_size_max:
        raise ValueError('Invalid learning-rate decay or rendering range')
    if any(n is not None and n < 1 for n in (config.max_steps, config.limit)):
        raise ValueError('Limits must be positive')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if config.device == 'auto' else config.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This trainer supports CPU and CUDA')
    if config.amp and device.type != 'cuda':
        raise ValueError('Mixed precision is supported only on CUDA')
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.set_num_threads(config.threads)
    if config.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    random.seed(config.seed); np.random.seed(config.seed % (2**32)); torch.manual_seed(config.seed)
    with run_lock(config.run_dir):
        checkpoints = Checkpoints(config.run_dir)
        if not config.resume and (checkpoints.latest.exists() or checkpoints.previous.exists()):
            raise FileExistsError('Checkpoints already exist; use --resume or a new --run-dir')
        identity = identity_for(config, device)
        saved = checkpoints.load(identity, compatible_code_hashes=(
            # Version before sample printing: identical training and checkpoint semantics.
            '3f297d0e0e09cb103d90d19d3a5544b975763a2f1b85190b7c81c6bec3e5312c',
        )) if config.resume else None
        vocabulary = Vocabulary.read(config.vocabulary)
        records, report = prepare_data(config, vocabulary)
        atomic_json(config.run_dir / 'font_coverage.json', report)
        atomic_json(config.run_dir / 'run_config.json', {
            'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in asdict(config).items()},
            'identity': identity, 'classes_including_blank': len(vocabulary),
        })
        print(f"Rows: {len(records):,}; rows with space replacements: {report['replaced_rows']:,}; device: {device}", flush=True)
        model = LineCNN(len(vocabulary), config.channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
        position = {'epoch': 0, 'next_batch': 0, 'step': 0}
        epoch_totals = empty_totals()
        log_offset = 0
        if saved:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            position = dict(saved['position'])
            epoch_totals = dict(saved['metrics']['epoch_totals'])
            log_offset = saved['metrics']['log_offset']
            restore_rng(saved['rng'])
            del saved
            print(f'Resuming: {position}', flush=True)
        metrics_log = MetricsLog(config.run_dir / 'metrics.jsonl', log_offset)
        model.train()

        def save():
            checkpoints.save({
                'identity': identity, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'position': dict(position), 'rng': rng_state(),
                'metrics': {'epoch_totals': dict(epoch_totals), 'log_offset': metrics_log.offset},
            })

        if not config.resume:
            save()  # Even interruption before the first periodic save has a restart point.
        total_batches = math.ceil(len(records) / config.batch_size)
        if not 0 <= position['next_batch'] < total_batches:
            raise ValueError('Invalid checkpoint batch position')
        if config.max_steps is not None and position['step'] >= config.max_steps:
            return position
        for epoch in range(position['epoch'], config.epochs):
            order = list(range(len(records)))
            random.Random(f'{config.seed}:order:{epoch}').shuffle(order)
            start = position['next_batch']
            remaining = order[start * config.batch_size:]
            dataset = EpochDataset(records, remaining, config, epoch)
            # Worker creation must never advance the model's global RNG.
            worker_generator = torch.Generator().manual_seed(config.seed + epoch)
            loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.workers, collate_fn=BatchRenderer(vocabulary),
                                generator=worker_generator,
                                multiprocessing_context='spawn' if config.workers else None,
                                timeout=120 if config.workers else 0)
            batch_started = time.perf_counter()
            for batch_index, batch in enumerate(loader, start=start):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config.amp):
                    logits = model(batch.images.to(device))
                # CTC and log-softmax run in float32 even when the CNN uses AMP.
                loss = ctc_loss(logits, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite CTC loss; restart from last checkpoint')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0,
                                                      error_if_nonfinite=not config.amp)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = not config.amp or scaler.get_scale() >= scale
                predictions = decode(logits, LineCNN.output_lengths(batch.image_widths), vocabulary)
                loss_value = float(loss.detach())
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                batch_seconds = time.perf_counter() - batch_started
                totals = batch_totals(predictions, batch.texts, loss_value, batch_seconds)
                add_totals(epoch_totals, totals)
                if updated:
                    position['step'] += 1
                position['next_batch'] = batch_index + 1
                end_epoch = position['next_batch'] == total_batches
                if end_epoch:
                    if updated or position['step'] > 0:
                        scheduler.step()
                    position['epoch'] = epoch + 1
                    position['next_batch'] = 0
                events = [dict(kind='batch', epoch=epoch + 1, batch=batch_index + 1,
                               step=position['step'], optimizer_updated=updated, **summary(totals))]
                if end_epoch:
                    epoch_summary = summary(epoch_totals)
                    events.append(dict(kind='epoch', epoch=epoch + 1, step=position['step'], **epoch_summary))
                    print(f"epoch={epoch + 1} accuracy={epoch_summary['exact_accuracy']:.2%} "
                          f"CER={epoch_summary['cer']:.2%} seconds={epoch_summary['seconds']:.3f}", flush=True)
                    epoch_totals = empty_totals()
                metrics_log.append(events)
                stop = config.max_steps is not None and position['step'] >= config.max_steps
                if end_epoch or stop or (updated and position['step'] % config.save_every == 0):
                    save()
                if updated and (position['step'] == 1 or position['step'] % config.log_every == 0):
                    print(f"step={position['step']} epoch={epoch + 1} loss={loss_value:.6f} accuracy={summary(totals)['exact_accuracy']:.2%} CER={summary(totals)['cer']:.2%} seconds={batch_seconds:.3f}", flush=True)
                    for reference, prediction in list(zip(batch.texts, predictions))[:config.log_samples]:
                        print(f'  正解: {reference!r}\n  予測: {prediction!r}', flush=True)
                if on_step is not None:
                    on_step(dict(position), batch.sample_ids)
                if stop:
                    return position
                batch_started = time.perf_counter()
        return position


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('text', 'vocabulary', 'font', 'run_dir'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, default=getattr(defaults, name))
    for name in ('batch_size', 'epochs', 'channels', 'seed', 'save_every', 'workers',
                 'font_size_min', 'font_size_max', 'threads', 'log_every', 'log_samples', 'max_steps', 'limit'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(defaults, name))
    for name in ('learning_rate', 'lr_decay'):
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=getattr(defaults, name))
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:N')
    for name in ('amp', 'deterministic', 'resume'):
        parser.add_argument('--' + name, action='store_true')
    config = TrainConfig(**vars(parser.parse_args()))
    try:
        train(config)
    except KeyboardInterrupt:
        print('Interrupted. Use --resume to restart from the last completed checkpoint.', flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
