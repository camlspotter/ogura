"""Greedy CTC accuracy, sample-weighted epoch totals and rollback-safe JSONL."""
import json
import math
from datetime import datetime
import time
import os
from pathlib import Path


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def worst_samples(references, predictions, count):
    """Rank this batch by per-sample CER descending; ties keep batch order."""
    if count <= 0:
        return []
    rows = [(reference, prediction, edit_distance(reference, prediction) / len(reference))
            for reference, prediction in zip(references, predictions)]
    return sorted(rows, key=lambda row: row[2], reverse=True)[:count]


def decode(logits, lengths, vocabulary):
    paths = logits.detach().argmax(dim=2).cpu().transpose(0, 1).tolist()
    texts = []
    for path, length in zip(paths, lengths.tolist()):
        characters = []
        last = None
        for token in path[:length]:
            if token != last and token != vocabulary.blank:
                characters.append(vocabulary.characters[token - 1])
            last = token
        texts.append(''.join(characters))
    return texts


def empty_totals():
    return dict(batches=0, samples=0, exact_matches=0, character_errors=0,
                reference_characters=0, loss_sum=0.0, seconds=0.0)


def batch_totals(predictions, references, loss, seconds):
    if len(predictions) != len(references) or not references:
        raise ValueError('Expected equally sized nonempty predictions and references')
    return dict(batches=1, samples=len(references),
                exact_matches=sum(a == b for a,b in zip(predictions,references)),
                character_errors=sum(edit_distance(a,b) for a,b in zip(references,predictions)),
                reference_characters=sum(map(len,references)),
                loss_sum=loss * len(references), seconds=seconds)


def add_totals(total, batch):
    for key in total:
        total[key] += batch[key]


def summary(totals):
    return {**totals, 'exact_accuracy': totals['exact_matches'] / totals['samples'],
            'cer': totals['character_errors'] / totals['reference_characters'],
            'mean_loss': totals['loss_sum'] / totals['samples'],
            'samples_per_second': totals['samples'] / totals['seconds'] if totals['seconds'] else 0.0}


class MetricsLog:
    """Logs are durable before checkpoints; uncommitted tails are discarded on resume."""
    def __init__(self, path, offset=0):
        self.path = Path(path)
        if offset and (not self.path.exists() or self.path.stat().st_size < offset):
            raise ValueError('metrics.jsonl is missing or shorter than the checkpoint requires')
        with self.path.open('r+b' if self.path.exists() else 'w+b') as stream:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())
        self.offset = offset

    def append(self, events):
        with self.path.open('ab') as stream:
            for event in events:
                stream.write((json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
            self.offset = stream.tell()


def epoch_eta(totals, total_batches, elapsed_seconds=None):
    """Estimate remaining training time from completed work and elapsed time."""
    completed = totals['batches']
    if completed == 0:
        return None
    if not 0 < completed <= total_batches:
        raise ValueError('Invalid epoch progress')
    elapsed = totals['seconds'] if elapsed_seconds is None else elapsed_seconds
    return elapsed / completed * (total_batches - completed)


def format_duration(seconds):
    if seconds is None:
        return '--:--:--'
    hours, remainder = divmod(math.ceil(max(0, seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def local_finish_time(remaining_seconds, now=None):
    if remaining_seconds is None:
        return '--'
    timestamp = time.time() if now is None else now
    return datetime.fromtimestamp(timestamp + max(0, remaining_seconds)).astimezone().isoformat(timespec='seconds')


def best_validation_results(best, log_path):
    """Return only metrics measured at the best checkpoint, never the last epoch."""
    if 'validation_results' in best:
        return best['validation_results']
    rows = []
    path = Path(log_path)
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if (event.get('epoch') == best['epoch'] and event.get('step') == best['step']
                        and event.get('kind') in ('validation', 'validation_baseline', 'validation_length')):
                    rows.append(event)
    if not any(row['kind'] == 'validation' for row in rows):
        rows.insert(0, dict(kind='validation', epoch=best['epoch'], step=best['step'], **best['metrics']))
    return rows


def print_best_validation(best, log_path):
    print(f"Best checkpoint: epoch={best['epoch']} step={best['step']}", flush=True)
    if "selection_score" in best:
        print(f"  selection={best['selection_metric']} CER={best['selection_score']:.4%}", flush=True)
    for row in best_validation_results(best, log_path):
        label = row['kind']
        if label == 'validation_length':
            label += f" dataset={row['dataset']} mode={row['mode']}"
        if 'font' in row:
            label += f" font={row['font']}"
        if 'length' in row:
            label += f" length={row['length']}"
        if 'exact_accuracy' not in row:
            continue
        print(f"  {label} accuracy={row['exact_accuracy']:.2%} CER={row['cer']:.4%}", flush=True)
