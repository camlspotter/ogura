"""Select a bounded pilot sample of local Wikipedia articles for Synth-JDoc."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from fontTools.ttLib import TTFont
import pyarrow.parquet as pq
from .synth_jdoc import ROOT, sha, write_json


def excerpt(text, budget):
    """Keep complete sentences/paragraphs where possible; no character substitution."""
    result = []
    remaining = budget
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if len(paragraph) < 40:
            continue
        if len(paragraph) > remaining:
            prefix = paragraph[:remaining]
            stop = prefix.rfind('。')
            paragraph = prefix[:stop+1] if stop >= 60 else prefix
        if paragraph:
            result.append(paragraph)
            remaining -= len(paragraph)
        if remaining < 80:
            break
    return result


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cmaps = []
    for path in args.font:
        with TTFont(path) as font:
            cmaps.append(set(font.getBestCmap()))
    supported = set.intersection(*cmaps)
    rng = random.Random(args.seed)
    files = sorted(args.source.glob('*.parquet'))
    if not files:
        raise ValueError('No local Parquet files found')
    candidates = []
    # Bounded reads: this pilot is not a uniform sample of the full corpus.
    for file in files:
        rows = []
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=['id','url','title','text']):
            rows.extend(batch.to_pylist())
            if len(rows) >= args.rows_per_shard:
                break
        rng.shuffle(rows)
        candidates.extend((file.name, row) for row in rows[:args.rows_per_shard])
    rng.shuffle(candidates)
    selected, used_ids, used_texts = [], set(), set()
    for shard, row in candidates:
        paragraphs = excerpt(row['text'], args.lengths[len(selected) % len(args.lengths)])
        joined = ''.join(paragraphs)
        title = row['title']
        digest = hashlib.sha256(joined.encode()).hexdigest()
        if len(joined) < 250 or len(title) > 35 or row['id'] in used_ids or digest in used_texts:
            continue
        if any(ord(c) not in supported for c in title + joined if not c.isspace()):
            continue
        if sum(bool(re.match(r'[\u3040-\u30ff\u4e00-\u9fff]', c)) for c in joined) / len(joined) < .5:
            continue
        selected.append(dict(id='wikipedia-ja-'+row['id'], title=title, paragraphs=paragraphs,
            source=dict(dataset='wikimedia/wikipedia', config=args.source.name, shard=shard,
                        article_id=row['id'], url=row['url'], title=title,
                        original_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                        modification='Excerpt of prose paragraphs; whitespace at paragraph edges removed')))
        used_ids.add(row['id']); used_texts.add(digest)
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError(f'Only {len(selected)} usable articles; increase --rows-per-shard')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in selected))
    provenance = args.source.parent/'source.json'
    write_json(args.output.with_suffix('.source.json'), dict(seed=args.seed, count=len(selected),
        sampling='bounded first rows per shard, shuffled, filtered for font coverage and Japanese prose',
        rows_per_shard=args.rows_per_shard, text_lengths=args.lengths, output_sha256=sha(args.output),
        fonts={str(p):sha(p) for p in args.font},
        upstream_source=json.loads(provenance.read_text()) if provenance.exists() else None))
    print(f'Wrote {len(selected)} distinct article excerpts to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parents[1]/'corpus/wikipedia/20231101.ja')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/synth-texts-120.jsonl')
    parser.add_argument('--font', type=Path, action='append', required=True)
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--rows-per-shard', type=int, default=128)
    parser.add_argument('--lengths', type=int, nargs='+', default=[400,700,1000,1400])
    args = parser.parse_args()
    if args.count < 1 or args.rows_per_shard < 1 or min(args.lengths) < 250:
        parser.error('count/rows-per-shard must be positive; lengths must be at least 250')
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under ogura/textdet')
    prepare(args)


if __name__ == '__main__':
    main()
