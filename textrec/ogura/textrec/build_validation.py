"""Extract fixed, held-out Wikipedia prose for line recognition validation."""
from ogura.textrec.paths import CORPUS_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import unicodedata

import pyarrow.parquet as pq

from ogura.textrec.diversity import fingerprints
from ogura.textrec.prose_filter import is_prose
from ogura.textrec.text_common import ROOT, LINE, digest, emit, split_article, strip_han_ivs


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def candidate(article, vocabulary, rng, forbidden, min_length=20, max_length=25):
    if not is_prose(article['text'], article['title']):
        return None
    lines = list(LINE.finditer(article['text']))
    rng.shuffle(lines)
    for line in lines:
        # Offsets refer to original code points; normalize only the selected span.
        if not is_prose(line.group(), article['title']):
            continue
        starts = list(range(0, max(0, len(line.group()) - min_length + 1), min_length))
        rng.shuffle(starts)
        for start in starts:
            end = min(start + max_length, len(line.group()))
            raw = line.group()[start:end]
            text = strip_han_ivs(raw)
            if not min_length <= len(text) <= max_length or not set(text) <= vocabulary:
                continue
            if text[0].isspace() or text[-1].isspace() or unicodedata.category(text[0]).startswith('M'):
                continue
            if (min_length >= 20 and not is_prose(text, article['title'])) or not forbidden.isdisjoint(fingerprints(text)):
                continue
            return dict(text=text, sample_id=digest(text), article_id=article['id'],
                        title=article['title'], url=article['url'],
                        start=line.start()+start, end=line.start()+end, split='validation')
    return None


def extract(articles, vocabulary, forbidden, count, seed, split_seed=20260915,
            min_length=20, max_length=25, excluded_articles=frozenset()):
    rng = random.Random(seed)
    pool = []
    eligible = 0
    capacity = max(count * 4, count + 500)
    for article in articles:
        if article['id'] in excluded_articles or split_article(article['id'], split_seed) != 'validation':
            continue
        row = candidate(article, vocabulary, rng, forbidden, min_length, max_length)
        if row is None:
            continue
        assert strip_han_ivs(article['text'][row['start']:row['end']]) == row['text']
        eligible += 1
        if len(pool) < capacity:
            pool.append(row)
        else:
            index = rng.randrange(eligible)
            if index < capacity:
                pool[index] = row
    rng.shuffle(pool)
    selected = []
    fragments = set()
    article_ids = set()
    for row in pool:
        current = fingerprints(row['text'])
        if row['article_id'] in article_ids or not fragments.isdisjoint(current):
            continue
        selected.append(row); fragments.update(current); article_ids.add(row['article_id'])
        if len(selected) == count:
            return selected, eligible
    raise ValueError(f'Only {len(selected)} eligible distinct examples; requested {count}')


def build(corpus, training_text, targets, output, count=1000, seed=20260915,
          min_length=20, max_length=25, exclude_validation=()):
    if not 1 <= min_length <= max_length:
        raise ValueError('Invalid length range')
    if count < 1:
        raise ValueError('Count must be positive')
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; reuse it or choose a new output directory')
    files = sorted(corpus.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'No Wikipedia parquet files: {corpus}')
    with targets.open(encoding='utf-8') as stream:
        vocabulary = {json.loads(line)['character'] for line in stream}
    forbidden = set()
    with training_text.open(encoding='utf-8') as stream:
        for text in stream:
            forbidden.update(fingerprints(text.removesuffix('\n')))
    excluded_articles = set()
    for path in exclude_validation:
        with path.open(encoding="utf-8") as stream:
            excluded_articles.update(json.loads(line)["article_id"] for line in stream)
    def articles():
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['id','url','title','text']):
                yield from batch.to_pylist()
            print(f'Scanned {path.name}', flush=True)
    rows, eligible = extract(articles(), vocabulary, forbidden, count, seed, min_length=min_length,
                             max_length=max_length, excluded_articles=excluded_articles)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.validation-', dir=output.parent))
    try:
        with (temporary/'validation.jsonl').open('w',encoding='utf-8') as js, (temporary/'validation.txt').open('w',encoding='utf-8') as txt:
            for row in rows:
                emit(js,row); txt.write(row['text']+'\n')
        shutil.copyfile(targets,temporary/'targets.jsonl')
        manifest = dict(samples=len(rows), eligible_articles=eligible, seed=seed, split_seed=20260915,
                        split='validation', one_sample_per_article=True, min_length=min_length, max_length=max_length,
                        prose_filter="parent line and article; excerpt also checked for lengths >=20",
                        excluded_validation_sha256=[sha256(p) for p in exclude_validation],
                        short_substrings_may_occur_in_training=min_length<16,
                        no_shared_16_character_fragments_with_training=True,
                        training_text_sha256=sha256(training_text), targets_sha256=sha256(targets),
                        validation_text_sha256=sha256(temporary/'validation.txt'),
                        corpus_files=[p.name for p in files], normalization='strip Han IVS only')
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.rename(output)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    print(f'Validation: {len(rows):,} samples -> {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=CORPUS_ROOT / 'wikipedia/20231101.ja')
    parser.add_argument('--training-text',type=Path,default=ROOT/'datasets/final_50_len20_25_hiragana_mix5/train.txt')
    parser.add_argument('--targets',type=Path,default=ROOT/'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/validation')
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20260915)
    parser.add_argument('--min-length',type=int,default=20)
    parser.add_argument('--max-length',type=int,default=25)
    parser.add_argument('--exclude-validation',type=Path,action='append',default=[],help='Exclude article IDs in this validation JSONL; repeatable')
    args=parser.parse_args()
    build(args.corpus,args.training_text,args.targets,args.output,args.count,args.seed,args.min_length,args.max_length,args.exclude_validation)


if __name__=='__main__':main()
