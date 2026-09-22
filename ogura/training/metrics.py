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


QUOTE_GROUPS = (frozenset('“”"'), frozenset("‘’'"))
QUOTE_CHARACTERS = frozenset().union(*QUOTE_GROUPS)


def same_quote_group(a, b):
    return any(a in group and b in group for group in QUOTE_GROUPS)


def weighted_edit_distance(reference, prediction):
    """Levenshtein cost with partial credit for quote variants and adjacent spaces.

    Some fonts (notably Kosugi Maru) leave substantial side bearings around
    quotation marks. These look like typed spaces even when the source has none.
    Exact source transcription remains preferable: insertion costs 0.5, not 0.
    Curly left/right and straight quotes can be distinct in some fonts but
    almost indistinguishable in others. Preserve their original code points:
    exact matches cost 0; substitutions within the double-quote group or within
    the single-quote group cost 0.5, regardless of font. Crossing these groups
    costs 1 for one-to-one substitutions. Two adjacent single quotes may look
    like one double quote: consume the pair versus one double quote at cost 0.5
    in either direction. Any variants within these groups qualify, but spaces
    between the single quotes do not. Extra quotes remain charged.
    We do not merge classifier classes or turn variants into full credit.
    Deletions, other substitutions, and spaces elsewhere still cost 1. Normalize
    label aliases before calling; do not strip or canonicalize quotation marks.

    This is evaluation only: CTC still learns the original single target string.
    If this partial-credit evaluation does not resolve the practical problem,
    extend the training loss to allow penalized optional spaces around quotes
    and penalized quote variants, including two-to-one pairs (e.g. weighted
    alternative targets),
    while preferring the original text. This would address visual ambiguity in
    learning itself; changing this evaluation score alone cannot change gradients.
    Do not simply add random spaces to labels: that teaches insertion as correct.
    """
    if reference == prediction:
        return 0.0
    left = []; last = ''
    for char in prediction:
        left.append(last)
        if char != ' ': last = char
    right = [''] * len(prediction); last = ''
    for j in range(len(prediction)-1, -1, -1):
        right[j] = last
        if prediction[j] != ' ': last = prediction[j]
    # Only a quote already present in the reference can justify partial credit.
    def insertion(i, j):
        if prediction[j] != ' ': return 1.0
        before = reference[i-1] if i else ''
        after = reference[i] if i < len(reference) else ''
        return .5 if ((before in QUOTE_CHARACTERS and same_quote_group(left[j], before)) or
                      (after in QUOTE_CHARACTERS and same_quote_group(right[j], after))) else 1.0
    previous = [0.0]
    for j in range(len(prediction)): previous.append(previous[-1]+insertion(0,j))
    two_back = None
    doubles, singles = QUOTE_GROUPS
    for i, char in enumerate(reference, 1):
        current = [float(i)]
        for j, other in enumerate(prediction, 1):
            current.append(min(previous[j]+1, current[-1]+insertion(i,j-1),
                               previous[j-1]+(0 if char == other else
                                              .5 if same_quote_group(char, other) else 1)))
            # Consume disjoint prefixes so no quote is counted twice or skipped.
            # Two retained rows support 2:1/1:2 matches in O(n*m) time/O(m) space.
            if i >= 2 and reference[i-2] in singles and char in singles and other in doubles:
                current[-1] = min(current[-1], two_back[j-1] + .5)
            if j >= 2 and char in doubles and prediction[j-2] in singles and other in singles:
                current[-1] = min(current[-1], previous[j-2] + .5)
        two_back, previous = previous, current
    return previous[-1]


def weighted_cer_label(row):
    return f" weighted_CER={row['weighted_cer']:.4%}" if 'weighted_cer' in row else ''


def worst_samples(references, predictions, count, evaluation_aliases=None):
    """Rank this batch by per-sample CER descending; ties keep batch order."""
    if count <= 0:
        return []
    references, predictions = evaluation_texts(references, predictions, evaluation_aliases)
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
        texts.append(vocabulary.aliases.normalize(''.join(characters)))
    return texts


def empty_totals():
    return dict(batches=0, samples=0, exact_matches=0, character_errors=0,
                reference_characters=0, loss_sum=0.0, seconds=0.0)


def evaluation_texts(references, predictions, aliases=None):
    """Normalize copies for scoring only; never changes CTC targets or images."""
    if aliases is None:
        return references, predictions
    return ([aliases.normalize(t) for t in references], [aliases.normalize(t) for t in predictions])


def batch_totals(predictions, references, loss, seconds, evaluation_aliases=None):
    if len(predictions) != len(references) or not references:
        raise ValueError('Expected equally sized nonempty predictions and references')
    references, predictions = evaluation_texts(references, predictions, evaluation_aliases)
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
        print(f"  {label} accuracy={row['exact_accuracy']:.2%} CER={row['cer']:.4%}{weighted_cer_label(row)}", flush=True)
