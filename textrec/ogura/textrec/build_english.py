"""Build reproducible English Wikipedia prose lines, without changing Japanese data."""
from ogura.textrec.paths import CORPUS_ROOT

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
from urllib.request import urlopen

import pyarrow.parquet as pq
from ogura.textrec.analyze_wikipedia import REVISION, REPO, ROOT
from ogura.textrec.diversity import fingerprints

CONFIG = '20231101.en'
# One pinned shard is enough for this initial supplement; not the whole English corpus.
SHARD = 'train-00000-of-00041.parquet'
SHA256 = '382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677'
SIZE = 420296449
LATIN = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿ]')
WORDS = re.compile(r'\S+')
FUNCTION = re.compile(r'\b(?:the|a|an|is|was|were|are|of|in|to|and|with|for|from|by)\b', re.I)
PAIRS = ('ff', 'fi', 'fl', 'ffi', 'ffl', 'll', 'ss', 'tt', 'ii', 'oo', 'ee', 'rr', 'nn', 'mm')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SHARD
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{CONFIG}/{SHARD}'
    if not path.exists():
        temp = path.with_suffix('.part')
        subprocess.run(['curl', '-fL', '--retry', '4', '--silent', '--show-error', '-o', str(temp), url], check=True)
        if temp.stat().st_size != SIZE or sha(temp) != SHA256:
            raise ValueError('Downloaded Wikipedia shard failed verification')
        temp.replace(path)
    if path.stat().st_size != SIZE or sha(path) != SHA256:
        raise ValueError(f'Wikipedia shard failed verification: {path}')
    return path, url


def prose(line):
    """Conservative paragraph filter, not a semantic language classifier."""
    words = WORDS.findall(line)
    return (len(words) >= 12 and len(line) >= 80 and
            len(LATIN.findall(line)) / len(line) >= .7 and
            FUNCTION.search(line) is not None and
            re.search(r'[.!?](?:["”\']|$| )', line) is not None and
            not re.search(r'https?://|www\.|[{}|=<>\\]|\bISBN\b', line) and
            not re.search(r'(?:\b[A-Za-z]\b\s+){5}', line))


def candidate(article, vocabulary, seed, low=20, high=25, required_character=None):
    """Return one literal, whole-word excerpt per article, with source offsets."""
    if re.search(r'^(?:List of|Lists of|Index of|Outline of|Glossary of)\b|alphabet|Unicode|disambiguation', article['title'], re.I):
        return None
    rng = random.Random(f'{seed}:{article["id"]}')
    paragraphs = [(m.start(), m.group()) for m in re.finditer(r'[^\n]+', article['text']) if (required_character is None or required_character in m.group()) and prose(m.group())]
    rng.shuffle(paragraphs)
    for offset, paragraph in paragraphs:
        tokens = list(WORDS.finditer(paragraph))
        starts = list(range(len(tokens)))
        rng.shuffle(starts)
        for i in starts:
            options = []
            for j in range(i+1, len(tokens)):
                start, end = tokens[i].start(), tokens[j].end()
                text = paragraph[start:end]
                if len(text) > high: break
                if len(text) < low: continue
                if ((required_character is None or required_character in text) and set(text) <= vocabulary and
                    all(c == ' ' or (ord(c) <= 255 and (c.isalpha() or c.isdigit())) or c in ",.;:!?'-\"()“”‘’–—" for c in text) and
                    len(LATIN.findall(text)) / len(text) >= .7 and
                    all(len(w.strip('.,;:!?\"()')) > 1 or w in ('a', 'A', 'I') for w in text.split())):
                    options.append(dict(text=text, article_id=article['id'], title=article['title'],
                                        url=article['url'], start=offset+start, end=offset+end))
            if options: return rng.choice(options)
    return None


def statistics(rows):
    texts = [r['text'] for r in rows]
    return dict(rows=len(rows), articles=len({r['article_id'] for r in rows}),
                characters=sum(map(len, texts)), lengths=dict(sorted(Counter(map(len, texts)).items())),
                character_counts=dict(sorted(Counter(''.join(texts)).items())),
                patterns={p: dict(occurrences=sum(len(re.findall('(?='+p+')', s)) for s in texts),
                                  rows=sum(p in s for s in texts)) for p in PAIRS})


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if args.train_count < 1 or args.validation_count < 1: raise ValueError('Counts must be positive')
    path, url = download(args.corpus)
    vocab = {json.loads(s)['character'] for s in args.targets.read_text().splitlines()}
    # Exclude overlap with the current Japanese training and every held-out length set.
    excluded = [args.training_text, *args.exclude_text]
    forbidden = set()
    for f in excluded:
        for line in f.read_text().splitlines(): forbidden.update(fingerprints(line))
    pools = {'train': [], 'validation': []}
    scanned = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['id','url','title','text']):
        for article in batch.to_pylist():
            scanned += 1
            row = candidate(article, vocab, args.seed)
            if row is None or not forbidden.isdisjoint(fingerprints(row['text'])): continue
            assert article['text'][row['start']:row['end']] == row['text']
            split = 'validation' if int(hashlib.sha256(f'{args.seed}:split:{article["id"]}'.encode()).hexdigest(),16) % 10 == 0 else 'train'
            pools[split].append(row)
    print(f'Scanned {scanned:,} articles; candidates: '+str({k:len(v) for k,v in pools.items()}), flush=True)
    rng = random.Random(args.seed)
    selected = {}
    # Validation first; no shared 16-character fragments, even across splits.
    for split, count in [('validation', args.validation_count), ('train', args.train_count)]:
        rng.shuffle(pools[split]); selected[split] = []
        for row in pools[split]:
            fp = fingerprints(row['text'])
            if not forbidden.isdisjoint(fp): continue
            selected[split].append(row); forbidden.update(fp)
            if len(selected[split]) == count: break
        if len(selected[split]) != count: raise ValueError(f'Insufficient distinct {split} examples')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.english-', dir=args.output.parent))
    try:
        for split, rows in selected.items():
            (temp/f'{split}.txt').write_text(''.join(r['text']+'\n' for r in rows))
            (temp/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        report = dict(source=dict(repository=REPO, revision=REVISION, config=CONFIG, shard=SHARD,
                                  sha256=SHA256, url=url, scope='single shard',
                                  license='Wikipedia text: CC BY-SA / GFDL; retain article attribution and consult source terms'),
                      seed=args.seed, length=[20,25], scanned_articles=scanned,
                      script_sha256=sha(__file__), targets_sha256=sha(args.targets),
                      excluded_files={str(p):sha(p) for p in excluded},
                      splits={k:statistics(v) for k,v in selected.items()},
                      output_sha256={p.name:sha(p) for p in sorted(temp.iterdir())})
        (temp/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temp.rename(args.output)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    print(f'Created {args.output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=CORPUS_ROOT / 'wikipedia'/CONFIG)
    p.add_argument('--output', type=Path, default=ROOT/'datasets/english_wikipedia_30k')
    p.add_argument('--targets', type=Path, default=ROOT/'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl')
    p.add_argument('--training-text', type=Path, default=ROOT/'datasets/final_50_len20_25_hiragana_mix5/train.txt')
    p.add_argument('--exclude-text', type=Path, action='append', default=None)
    p.add_argument('--train-count', type=int, default=30000)
    p.add_argument('--validation-count', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260920)
    args = p.parse_args()
    if args.exclude_text is None:
        args.exclude_text = [ROOT/'datasets'/d/'validation.txt' for d in ('validation','validation_short5','validation_long80')]
    build(args)


if __name__ == '__main__': main()
