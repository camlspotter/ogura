"""Train a line CNN with on-the-fly rendering and recoverable checkpoints.

Run: python -m ogura.training.train --help
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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
from .checkpoint import Checkpoints, restore_rng, rng_state, write_best
from .evaluate import evaluate
from .model import LineCNN, ctc_loss, make_model
from .metrics import MetricsLog, add_totals, batch_totals, decode, empty_totals, summary, worst_samples, print_best_validation, epoch_eta, format_duration, local_finish_time
from .render import BatchRenderer, Vocabulary, font_characters, parameters_for_sample, replace_unsupported


@dataclass(frozen=True)
class TrainConfig:
    text: Path = ROOT / 'datasets/final_50_len20_25_hiragana_mix5/train.txt'
    vocabulary: Path = ROOT / 'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl'
    font: Path = ROOT / 'corpus/fonts/NotoSansCJKjp-Regular.otf'
    extra_fonts: tuple[Path, ...] = ()
    init_from: Path | None = None
    padding_min: int = 4
    padding_max: int = 4
    vertical_jitter: int = 0
    vertical_full_range: bool = False
    clean_probability: float = 0.0
    selection_metric: str = "validation-cer"
    validation_augmented: bool = False
    run_dir: Path = ROOT / 'runs/noto48'
    validation_text: Path | None = None
    validation_font_size: int = 40
    monitor_validation: tuple[Path, ...] = ()
    device: str = 'auto'
    batch_size: int = 32
    epochs: int = 10
    early_stopping_patience: int = 0
    learning_rate: float = 0.001
    lr_decay: float = 0.95
    channels: int = 32
    model_type: str = "small"
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
    fonts = (config.font, *config.extra_fonts)
    supported = set.intersection(*(set(font_characters(str(p.resolve()))) for p in fonts))
    if any(ord(' ') not in font_characters(str(p.resolve())) for p in fonts):
        raise ValueError('Every font must support replacement space U+0020')
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
            records.append((text if config.extra_fonts else replace_unsupported(text, str(config.font.resolve())), sample_id))
    if not records:
        raise ValueError('Text dataset is empty')
    report = {
        'total_rows': total, 'unchanged_rows': total - len(replaced_lines), 'excluded_rows': 0,
        'replacement': 'U+0020',
        'coverage_scope': 'unsupported in at least one font; actual replacements depend on sampled font',
        'replaced_rows': len(replaced_lines),
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
            tuple(p.resolve() for p in self.config.extra_fonts), self.config.padding_min,
            self.config.padding_max, self.config.vertical_jitter, self.config.clean_probability, self.config.vertical_full_range,
        ), sample_id)


def identity_for(config, device):
    # Paths and operational settings may differ after copying a run to another machine.
    settings = asdict(config)
    for key in ('text', 'vocabulary', 'font', 'run_dir', 'validation_text', 'device', 'resume', 'max_steps',
                'workers', 'save_every', 'log_every', 'log_samples', 'epochs', 'extra_fonts', 'init_from', 'monitor_validation', 'early_stopping_patience'):
        settings.pop(key)
    if config.selection_metric == "validation-cer":
        settings.pop("selection_metric")  # Preserve legacy resume identities.
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        code.update(path.name.encode()); code.update(path.read_bytes())
    return {
        **({'selection_validation_sha256': sorted(sha256(p) for p in config.monitor_validation)}
           if config.selection_metric == 'mean-augmented-cer' else {}),
        'target_normalization': 'unsupported-to-space-collapse-ascii-v1',
        'format': 3, 'text_sha256': sha256(config.text),
        'validation_sha256': sha256(config.validation_text) if config.validation_text else None,
        'vocabulary_sha256': sha256(config.vocabulary), 'font_sha256': sha256(config.font),
        'extra_font_sha256': [sha256(p) for p in config.extra_fonts],
        'training_code_sha256': code.hexdigest(), 'settings': settings,
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'pillow': PIL.__version__, 'fonttools': fontTools.__version__,
                    'freetype': features.version('freetype2'), 'device_type': device.type},
    }


def load_initial_weights(model, path, vocabulary, channels, model_type="small", vocabulary_path=None):
    """Load weights only; verify labels and architecture for either export format."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'characters' in state:
        labels_match = state['characters'] == list(vocabulary.characters)
        settings = state
    else:
        settings = state.get('identity', {}).get('settings', {})
        labels_match = (vocabulary_path is not None and
                        state.get('identity', {}).get('vocabulary_sha256') == sha256(vocabulary_path))
    if not labels_match or settings.get('channels') != channels:
        raise ValueError('Initial model vocabulary/order or channels differ from this run')
    if settings.get('model_type', 'small') != model_type:
        raise ValueError('Initial model architecture differs from this run')
    model.load_state_dict(state['model'], strict=True)


def selection_score(metric, validation, events):
    if metric == 'validation-cer':
        return validation['cer']
    rows = [e for e in events if e['kind'] == 'validation_length' and e['mode'] == 'augmented']
    if sorted((e['min_length'], e['max_length']) for e in rows) != [(5, 5), (80, 80)]:
        raise ValueError('Mean selection requires augmented results for both 5 and 80 characters')
    return math.fsum([validation['cer']] + [e['cer'] for e in rows]) / 3


def train(config: TrainConfig, on_step=None):
    """on_step is an optional observer called only after a complete update."""
    if config.selection_metric not in ('validation-cer', 'mean-augmented-cer'):
        raise ValueError('Unknown selection metric')
    if config.selection_metric == 'mean-augmented-cer' and (
            not config.validation_augmented or not config.validation_text or len(config.monitor_validation) != 2):
        raise ValueError('mean-augmented-cer requires augmented main validation and two monitors (5 and 80 characters)')
    positive = (config.batch_size, config.epochs, config.channels, config.save_every,
                config.threads, config.log_every, config.validation_font_size)
    if min(positive) < 1 or config.log_samples < 0 or config.workers < 0 or not config.learning_rate > 0:
        raise ValueError('Invalid training configuration')
    if not 0 < config.lr_decay <= 1 or not 1 <= config.font_size_min <= config.font_size_max:
        raise ValueError('Invalid learning-rate decay or rendering range')
    if any(n is not None and n < 1 for n in (config.max_steps, config.limit)):
        raise ValueError('Limits must be positive')
    if config.validation_augmented and not config.validation_text:
        raise ValueError('--validation-augmented requires --validation-text')
    if config.early_stopping_patience < 0:
        raise ValueError('Early stopping patience must be nonnegative')
    if config.early_stopping_patience and not config.validation_text:
        raise ValueError('Early stopping requires --validation-text')
    if config.resume and config.init_from:
        raise ValueError('--resume and --init-from are mutually exclusive')
    parameters_for_sample(config.font, config.seed, 0, 'check', config.font_size_min,
                          config.font_size_max, config.extra_fonts, config.padding_min,
                          config.padding_max, config.vertical_jitter, config.clean_probability, config.vertical_full_range)
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
            # Best-score reporting does not change training or validation.
            '35d5408dcd10838a5cb7e0d95a188a4fd46c14cd2c098a38abf460987ea96121',
            # The original small model is unchanged by the optional residual architecture.
            '7707918703b7974d4497433b68e6253d37f75bafb005a2e98ca690a164c44a1a',
            '6be33a35ce25e831be7544cc45f0e4f632eab9438c2be3185bc6f9c804af3041',
            '74af5387661339582ec621527686e1a14ad01cdc67a2f7b4a3948ec236c751df',
            'e9956d6bc8c4b393adb9a770635a61f2f2c7ae7f38763067d5fe9aaa6bf347ca',
        )) if config.resume else None
        vocabulary = Vocabulary.read(config.vocabulary)
        records, report = prepare_data(config, vocabulary)
        validation_dataset = None
        augmented_validation_dataset = None
        if config.validation_text:
            validation_config = replace(config, text=config.validation_text, limit=None,
                                        font_size_min=config.validation_font_size,
                                        font_size_max=config.validation_font_size, extra_fonts=(),
                                        padding_min=4, padding_max=4, vertical_jitter=0, clean_probability=0, vertical_full_range=False)
            validation_records, validation_report = prepare_data(validation_config, vocabulary)
            normalized_training = {replace_unsupported(text, str(config.font.resolve())) for text,_ in records}
            if normalized_training & {text for text,_ in validation_records}:
                raise ValueError('Validation text overlaps training text after font replacement')
            manifest_path = config.validation_text.parent / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                expected = {'training_text_sha256': identity['text_sha256'],
                            'targets_sha256': identity['vocabulary_sha256'],
                            'validation_text_sha256': identity['validation_sha256'], 'split': 'validation'}
                if any(manifest.get(k) != value for k,value in expected.items()):
                    raise ValueError('Validation manifest does not match this training dataset')
            validation_dataset = EpochDataset(validation_records, list(range(len(validation_records))),
                                              validation_config, epoch=0)
            if config.validation_augmented:
                augmented_config = replace(config, text=config.validation_text, limit=None, clean_probability=0)
                augmented_records, _ = prepare_data(augmented_config, vocabulary)
                augmented_validation_dataset = EpochDataset(augmented_records, list(range(len(augmented_records))),
                                                            augmented_config, epoch=0)
            atomic_json(config.run_dir / 'validation_font_coverage.json', validation_report)
        from ogura.evaluate_lengths import validation_sets, evaluate_sets
        monitor_datasets = validation_sets(config, vocabulary, config.monitor_validation, identity)
        if config.selection_metric == 'mean-augmented-cer':
            ranges = sorted((info['min_length'], info['max_length']) for info, _ in monitor_datasets
                            if info['mode'] == 'augmented')
            if ranges != [(5, 5), (80, 80)]:
                raise ValueError('Mean selection requires exactly 5-character and 80-character monitors')
            if not all(20 <= len(text) <= 25 for text in config.validation_text.read_text(encoding='utf-8').splitlines()):
                raise ValueError('Mean selection requires main validation lengths of 20 to 25')
        atomic_json(config.run_dir / 'font_coverage.json', report)
        atomic_json(config.run_dir / 'run_config.json', {
            'arguments': {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k in ('extra_fonts', 'monitor_validation') else v for k,v in asdict(config).items()},
            'identity': identity, 'classes_including_blank': len(vocabulary),
        })
        print(f"Rows: {len(records):,}; rows potentially needing space replacements: {report['replaced_rows']:,}; device: {device}", flush=True)
        model = make_model(len(vocabulary), config.channels, config.model_type).to(device)
        if config.init_from:
            load_initial_weights(model, config.init_from, vocabulary, config.channels, config.model_type, config.vocabulary)
            print(f'Initialized weights from: {config.init_from}; optimizer and epoch start fresh', flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
        position = {'epoch': 0, 'next_batch': 0, 'step': 0}
        epoch_totals = empty_totals()
        log_offset = 0
        epoch_elapsed = 0.0
        segment_started = None
        segment_elapsed = 0.0
        best = None
        if saved:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            position = dict(saved['position'])
            epoch_totals = dict(saved['metrics']['epoch_totals'])
            log_offset = saved['metrics']['log_offset']
            epoch_elapsed = saved['metrics'].get('epoch_elapsed_seconds', epoch_totals['seconds'])
            best = saved['best']
            restore_rng(saved['rng'])
            del saved
            print(f'Resuming: {position}', flush=True)
        metrics_log = MetricsLog(config.run_dir / 'metrics.jsonl', log_offset)
        write_best(config.run_dir, best)
        model.train()

        def save():
            elapsed_at_save = (segment_elapsed + time.perf_counter() - segment_started
                               if segment_started is not None else epoch_elapsed)
            checkpoints.save({
                'identity': identity, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'position': dict(position), 'rng': rng_state(),
                'metrics': {'epoch_totals': dict(epoch_totals), 'log_offset': metrics_log.offset,
                            'epoch_elapsed_seconds': elapsed_at_save},
                'best': best,
            })

            write_best(config.run_dir, best)

        def validate_and_select(epoch_number):
            nonlocal best
            events = []
            validation = None
            if validation_dataset is not None:
                validation = evaluate(model, validation_dataset, vocabulary, config.batch_size, device)
                if augmented_validation_dataset is not None:
                    events.append(dict(kind='validation_baseline', epoch=epoch_number,
                                       step=position['step'], **validation))
                    print(f"validation_baseline epoch={epoch_number} accuracy={validation['exact_accuracy']:.2%} CER={validation['cer']:.2%}", flush=True)
                    validation = evaluate(model, augmented_validation_dataset, vocabulary, config.batch_size, device)
                events.append(dict(kind='validation', epoch=epoch_number, step=position['step'], **validation))
            if monitor_datasets:
                for result in evaluate_sets(model, monitor_datasets, vocabulary, config.batch_size, device):
                    events.append(dict(kind='validation_length', epoch=epoch_number, step=position['step'], **result))
            if validation is not None:
                score = selection_score(config.selection_metric, validation, events)
                improved = best is None or score < best.get('selection_score', best['metrics']['cer'])
                next(e for e in events if e['kind'] == 'validation')['is_best'] = improved
                if improved:
                    best = dict(identity=identity, epoch=epoch_number, step=position['step'],
                                metrics=validation, characters=list(vocabulary.characters),
                                channels=config.channels, model_type=config.model_type,
                                selection_metric=config.selection_metric, selection_score=score,
                                validation_results=[dict(e) for e in events],
                                model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
                print(f"validation epoch={epoch_number} accuracy={validation['exact_accuracy']:.2%} "
                      f"CER={validation['cer']:.2%} seconds={validation['seconds']:.3f} best={improved}", flush=True)
                if config.selection_metric == 'mean-augmented-cer':
                    events.append(dict(kind='selection', epoch=epoch_number, step=position['step'],
                                       metric=config.selection_metric, cer=score, is_best=improved))
                    print(f"selection epoch={epoch_number} mean_augmented_CER={score:.4%} best={improved}", flush=True)
            return events

        def early_stop_event():
            # Reconstruct from committed epochs, including checkpoints predating early stopping.
            if not config.early_stopping_patience or best is None:
                return None
            stale_epochs = position['epoch'] - best['epoch']
            if stale_epochs < config.early_stopping_patience:
                return None
            return dict(kind='early_stop', epoch=position['epoch'], step=position['step'],
                        patience=config.early_stopping_patience, stale_epochs=stale_epochs,
                        best_epoch=best['epoch'], best_cer=best.get('selection_score', best['metrics']['cer']),
                        selection_metric=config.selection_metric)

        def announce_early_stop(event):
            print(f"Early stopping: no {config.selection_metric} improvement for {event['stale_epochs']} epochs "
                  f"(patience={event['patience']}); best epoch={event['best_epoch']} "
                  f"CER={event['best_cer']:.4%}", flush=True)
            print_best_validation(best, config.run_dir / 'metrics.jsonl')

        if not config.resume:
            if config.init_from and config.selection_metric == 'mean-augmented-cer':
                metrics_log.append(validate_and_select(0))
            save()  # Even interruption before the first periodic save has a restart point.
        total_batches = math.ceil(len(records) / config.batch_size)
        if not 0 <= position['next_batch'] < total_batches:
            raise ValueError('Invalid checkpoint batch position')
        if config.max_steps is not None and position['step'] >= config.max_steps:
            return position
        already_stopped = early_stop_event()
        if already_stopped:
            announce_early_stop(already_stopped)
            return position
        for epoch in range(position['epoch'], config.epochs):
            segment_elapsed = epoch_elapsed
            segment_started = time.perf_counter()
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
                elapsed = segment_elapsed + time.perf_counter() - segment_started
                eta = epoch_eta(epoch_totals, total_batches, elapsed)
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
                          f"CER={epoch_summary['cer']:.2%} seconds={epoch_summary['seconds']:.3f} elapsed={format_duration(elapsed)}", flush=True)
                    epoch_totals = empty_totals()
                    epoch_elapsed = 0.0
                    segment_started = None
                if end_epoch:
                    events.extend(validate_and_select(epoch + 1))
                stopping_event = early_stop_event() if end_epoch else None
                if stopping_event:
                    events.append(stopping_event)
                metrics_log.append(events)
                stop = bool(stopping_event) or config.max_steps is not None and position['step'] >= config.max_steps
                if end_epoch or stop or (updated and position['step'] % config.save_every == 0):
                    save()
                if updated and (position['step'] == 1 or position['step'] % config.log_every == 0):
                    if not end_epoch:
                        elapsed = segment_elapsed + time.perf_counter() - segment_started
                        eta = epoch_eta(epoch_totals, total_batches, elapsed)
                    print(f"step={position['step']} epoch={epoch + 1} loss={loss_value:.6f} accuracy={summary(totals)['exact_accuracy']:.2%} CER={summary(totals)['cer']:.2%} seconds={batch_seconds:.3f} batch={batch_index + 1}/{total_batches} ({(batch_index + 1) / total_batches:.1%}) elapsed={format_duration(elapsed)} remaining={format_duration(eta)} finish_local={local_finish_time(eta)}", flush=True)
                    for reference, prediction, sample_cer in worst_samples(batch.texts, predictions, config.log_samples):
                        print(f'  CER: {sample_cer:.2%}\n  正解: {reference!r}\n  予測: {prediction!r}', flush=True)
                if on_step is not None:
                    on_step(dict(position), batch.sample_ids)
                if stop:
                    if stopping_event:
                        announce_early_stop(stopping_event)
                    return position
                batch_started = time.perf_counter()
        return position


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('text', 'vocabulary', 'font', 'run_dir', 'validation_text', 'init_from'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, default=getattr(defaults, name))
    for name in ('batch_size', 'epochs', 'early_stopping_patience', 'channels', 'seed', 'save_every', 'workers',
                 'font_size_min', 'font_size_max', 'padding_min', 'padding_max', 'vertical_jitter', 'validation_font_size', 'threads', 'log_every', 'log_samples', 'max_steps', 'limit'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(defaults, name))
    for name in ('learning_rate', 'lr_decay', 'clean_probability'):
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=getattr(defaults, name))
    parser.add_argument('--selection-metric', choices=('validation-cer', 'mean-augmented-cer'), default='validation-cer')
    parser.add_argument('--model-type', choices=('small', 'residual'), default='small')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:N')
    parser.add_argument('--monitor-validation', type=Path, action='append', default=[])
    parser.add_argument('--extra-font', dest='extra_fonts', type=Path, action='append', default=[])
    for name in ('amp', 'deterministic', 'resume', 'validation-augmented', 'vertical-full-range'):
        parser.add_argument('--' + name, action='store_true')
    args = vars(parser.parse_args())
    args['extra_fonts'] = tuple(args['extra_fonts'])
    args['monitor_validation'] = tuple(args['monitor_validation'])
    config = TrainConfig(**args)
    try:
        train(config)
    except KeyboardInterrupt:
        print('Interrupted. Use --resume to restart from the last completed checkpoint.', flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
