"""Fill per-character deficits with seeded random hiragana and multiple distinct targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

from ogura.text_common import digest, emit, export_plaintext

HIRAGANA = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ'


def generate(base, output, goal=50, seed=20260915, min_length=20, targets_per_line=5):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    with (base/'targets.jsonl').open(encoding='utf-8') as source:
        targets = [json.loads(line)['character'] for line in source]
    vocab = set(targets)
    alphabet = ''.join(c for c in HIRAGANA if c in vocab)
    if not alphabet or not 1 <= min_length <= 25 or goal < 1 or not 1 <= targets_per_line <= min_length or len(alphabet) <= targets_per_line:
        raise ValueError('Invalid generator configuration')
    rows = []
    counts = Counter()
    seen = set()
    with (base/'train.jsonl').open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            assert not row.get('synthetic'), 'Base must contain natural examples only'
            text = row['text']
            assert min_length <= len(text) <= 25 and set(text) <= vocab
            assert row['sample_id'] == digest(text) and row['sample_id'] not in seen
            seen.add(row['sample_id']); counts.update(set(text)); rows.append(row)
    natural_counts = counts.copy()
    natural_total = len(rows)
    rng = random.Random(seed)
    generated = Counter()
    output.mkdir(parents=True)
    active = [c for c in targets if counts[c] < goal]
    indices = {c: i for i, c in enumerate(active)}
    with (output/'synthetic.jsonl').open('w', encoding='utf-8') as synthetic:
        while active:
            selected = rng.sample(active, min(targets_per_line, len(active)))
            length = rng.randint(min_length, 25)
            filler = ''.join(c for c in alphabet if c not in selected)
            text = list(rng.choices(filler, k=length))
            positions = rng.sample(range(length), len(selected))
            for char, position in zip(selected, positions):
                text[position] = char
            text = ''.join(text)
            key = digest(text)
            if key in seen:
                continue
            row = dict(sample_id=key, text=text, length=length, synthetic=True,
                       generation='random_hiragana',
                       embedded_targets=[dict(character=c, position=p) for c, p in zip(selected, positions)])
            emit(synthetic, row)
            rows.append(row); seen.add(key); counts.update(set(text)); generated.update(selected)
            for char in sorted(set(text)):
                if char in indices and counts[char] >= goal:
                    index = indices.pop(char)
                    last = active.pop()
                    if index < len(active):
                        active[index] = last
                        indices[last] = index
    random.Random(seed).shuffle(rows)
    with (output/'train.jsonl').open('w', encoding='utf-8') as merged:
        for row in rows:
            emit(merged, row)
    shutil.copyfile(base/'targets.jsonl', output/'targets.jsonl')
    metadata = dict(synthesis='random_hiragana', targets_per_line=targets_per_line, target_samples_per_entry=goal,
                    min_tokens=min_length, max_tokens=25, seed=seed, hiragana=alphabet,
                    natural_base=str(base.resolve()),
                    natural_sha256=hashlib.sha256((base/'train.jsonl').read_bytes()).hexdigest(),
                    target_sha256=hashlib.sha256((base/'targets.jsonl').read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    summary = dict(samples=len(rows), natural_samples=natural_total,
                   synthetic_samples=len(rows)-natural_total, target_entries=len(targets),
                   supplemented_characters=len(generated))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (output/'coverage.jsonl').open('w', encoding='utf-8') as coverage:
        for char in targets:
            emit(coverage, dict(character=char, natural_count=natural_counts[char],
                                generated_for_character=generated[char], sample_count=counts[char]))
    # Verify written output independently, including provenance-free generation.
    actual = Counter(); ids = set(); synthetic_total = 0
    with (output/'train.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); text = row['text']
            assert min_length <= len(text) <= 25 and set(text) <= vocab
            assert not any(c in text for c in '\r\n\x85\u2028\u2029')
            assert row['sample_id'] == digest(text) and row['sample_id'] not in ids
            ids.add(row['sample_id']); actual.update(set(text))
            if row.get('synthetic'):
                synthetic_total += 1
                embedded = row['embedded_targets']
                chars = [e['character'] for e in embedded]
                positions = [e['position'] for e in embedded]
                assert 1 <= len(chars) <= targets_per_line
                assert len(set(chars)) == len(chars) and len(set(positions)) == len(positions)
                for char, pos in zip(chars, positions):
                    assert text[pos] == char and text.count(char) == 1
                assert set(c for i,c in enumerate(text) if i not in positions) <= set(alphabet)
                assert 'article_id' not in row and 'base_text' not in row
    assert actual == counts and all(actual[c] >= goal for c in targets)
    assert synthetic_total == summary['synthetic_samples']
    reports = output/'reports'; reports.mkdir()
    (reports/'verification.json').write_text(json.dumps(dict(
        all_samples_verified=len(ids), synthetic_samples_verified=synthetic_total,
        all_targets_meet_goal=True, natural_source_verification=str(base/'reports/verification.json')
    ), indent=2)+'\n')
    export_plaintext(output)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--goal', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--min-length', type=int, default=20)
    parser.add_argument('--targets-per-line', type=int, default=5)
    args = parser.parse_args()
    generate(args.base, args.output, args.goal, args.seed, args.min_length, args.targets_per_line)


if __name__ == '__main__':
    main()
