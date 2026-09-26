"""Mix Japanese and English training text and bind held-out sets to the mixture."""
import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

from ogura.textrec.analyze_wikipedia import ROOT
from ogura.textrec.build_english import sha
from ogura.textrec.diversity import fingerprints


def prepare(japanese, english, output, seed=20260920):
    if output.exists(): raise FileExistsError(output)
    targets = japanese/'targets.jsonl'
    vocabulary = {json.loads(s)['character'] for s in targets.read_text().splitlines()}
    provenance = json.loads((english/'manifest.json').read_text())
    if provenance['targets_sha256'] != sha(targets): raise ValueError('English vocabulary mismatch')
    source_train = japanese/'train.txt'
    if sha(source_train) not in provenance['excluded_files'].values():
        raise ValueError('English extraction did not exclude this Japanese training text')
    for name, expected in provenance['output_sha256'].items():
        if sha(english/name) != expected: raise ValueError(f'English source changed: {name}')
    groups = [('english', english/'validation.txt')]
    groups += [(name, ROOT/'datasets'/name/'validation.txt') for name in ('validation','validation_short5','validation_long80')]
    japanese_rows = source_train.read_text().splitlines()
    english_rows = (english/'train.txt').read_text().splitlines()
    rows = japanese_rows + english_rows
    if any(not s or not set(s) <= vocabulary for s in rows): raise ValueError('Invalid training text')
    if len(set(rows)) != len(rows): raise ValueError('Duplicate training rows')
    forbidden = set()
    for text in rows: forbidden.update(fingerprints(text))
    for name, path in groups:
        if name != 'english':
            old = json.loads((path.parent/'manifest.json').read_text())
            expected = dict(training_text_sha256=sha(source_train), targets_sha256=sha(targets),
                            validation_text_sha256=sha(path), split='validation')
            if any(old.get(k) != v for k,v in expected.items()): raise ValueError(f'Invalid original validation manifest: {path}')
            if sha(path) not in provenance['excluded_files'].values(): raise ValueError('English extraction did not exclude validation')
        for text in path.read_text().splitlines():
            if not text or not set(text) <= vocabulary: raise ValueError(f'Invalid validation labels: {path}')
            if not forbidden.isdisjoint(fingerprints(text)): raise ValueError(f'Validation overlaps mixed training text: {path}')
    random.Random(seed).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.mixed-', dir=output.parent))
    try:
        (temp/'train.txt').write_text(''.join(s+'\n' for s in rows))
        shutil.copyfile(targets, temp/'targets.jsonl')
        for name, path in groups:
            directory = temp/name; directory.mkdir()
            shutil.copyfile(path, directory/'validation.txt')
            shutil.copyfile(targets, directory/'targets.jsonl')
            if (path.parent/'validation.jsonl').exists(): shutil.copyfile(path.parent/'validation.jsonl', directory/'validation.jsonl')
            lengths = [len(s) for s in path.read_text().splitlines()]
            manifest = dict(split='validation', samples=len(lengths), min_length=min(lengths), max_length=max(lengths),
                            training_text_sha256=sha(temp/'train.txt'), targets_sha256=sha(targets),
                            validation_text_sha256=sha(path), source_text=str(path),
                            source_manifest_sha256=sha(path.parent/'manifest.json'))
            (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        manifest = dict(seed=seed, japanese_rows=len(japanese_rows), english_rows=len(english_rows),
                        total_rows=len(rows), training_text_sha256=sha(temp/'train.txt'),
                        japanese_text_sha256=sha(source_train), english_manifest_sha256=sha(english/'manifest.json'),
                        targets_sha256=sha(targets), script_sha256=sha(__file__),
                        validation_sets=[name for name,_ in groups])
        (temp/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        temp.rename(output)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    print(f'Created {output}: {len(japanese_rows):,} Japanese + {len(english_rows):,} English = {len(rows):,} rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--japanese', type=Path, default=ROOT/'datasets/final_50_len20_25_hiragana_mix5')
    parser.add_argument('--english', type=Path, default=ROOT/'datasets/english_wikipedia_30k')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/japanese_english_30k')
    parser.add_argument('--seed', type=int, default=20260920)
    args=parser.parse_args()
    prepare(args.japanese, args.english, args.output, args.seed)


if __name__ == '__main__': main()
