"""Extract held-out natural quotation-mark examples from the pinned English shard."""
from ogura.textrec.paths import CORPUS_ROOT

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import pyarrow.parquet as pq
from ogura.textrec.build_english import ROOT, CONFIG, download, candidate, statistics, sha, REVISION, SHARD
from ogura.textrec.diversity import fingerprints

QUOTES = '“”‘’'


def select(pool, goal, forbidden, seed):
    rng=random.Random(seed);rng.shuffle(pool)
    counts=Counter();rows=[];used=set()
    while min(counts[c] for c in QUOTES)<goal:
        c=min((c for c in QUOTES if counts[c]<goal),key=lambda c:sum(c in r['text'] and r['article_id'] not in used for r in pool))
        found=False
        for r in pool:
            if c not in r['text'] or r['article_id'] in used:continue
            fp=fingerprints(r['text'])
            if not forbidden.isdisjoint(fp):continue
            rows.append(r);used.add(r['article_id']);forbidden.update(fp)
            counts.update(set(r['text']) & set(QUOTES));found=True
            if counts[c]>=goal:break
        if not found:raise ValueError(f'Insufficient distinct examples for {c}: {dict(counts)}')
    return rows


def build(args):
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.train_per_character,args.validation_per_character)<1:raise ValueError('Counts must be positive')
    source,url=download(CORPUS_ROOT / 'wikipedia'/CONFIG)
    vocabulary={json.loads(s)['character'] for s in args.targets.read_text().splitlines()}
    if not set(QUOTES)<=vocabulary:raise ValueError('Quote characters missing from targets')
    english=ROOT/'datasets/english_wikipedia_30k'
    used=set();inputs=[args.targets]
    for split in ('train','validation'):
        p=english/f'{split}.jsonl';inputs.append(p)
        used.update(json.loads(s)['article_id'] for s in p.read_text().splitlines())
    forbidden=set()
    for p in [ROOT/'datasets/japanese_english_30k/train.txt', english/'validation.txt',
              *[ROOT/'datasets'/d/'validation.txt' for d in ('validation','validation_short5','validation_long80')]]:
        inputs.append(p)
        for s in p.read_text().splitlines():forbidden.update(fingerprints(s))
    pools={'train':[],'validation':[]};scanned=0
    for b in pq.ParquetFile(source).iter_batches(batch_size=256,columns=['id','title','url','text']):
        for a in b.to_pylist():
            scanned+=1
            if a['id'] in used or not set(a['text']) & set(QUOTES):continue
            split='validation' if int(hashlib.sha256(f'{args.seed}:split:{a["id"]}'.encode()).hexdigest(),16)%10==0 else 'train'
            seen=set()
            for c in QUOTES:
                if c not in a['text']:continue
                r=candidate(a,vocabulary,args.seed,required_character=c)
                if r and r['text'] not in seen and forbidden.isdisjoint(fingerprints(r['text'])):
                    assert a['text'][r['start']:r['end']]==r['text']
                    pools[split].append(r);seen.add(r['text'])
        if scanned%25600==0:print(f'Scanned {scanned:,} articles',flush=True)
    selected={}
    for split,goal in [('validation',args.validation_per_character),('train',args.train_per_character)]:
        selected[split]=select(pools[split],goal,forbidden,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.quotes-',dir=args.output.parent))
    try:
        shutil.copyfile(args.targets,temp/'targets.jsonl')
        for split,rows in selected.items():
            (temp/f'{split}.txt').write_text(''.join(r['text']+'\n' for r in rows))
            (temp/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        report=dict(source=dict(revision=REVISION,config=CONFIG,shard=SHARD,sha256=sha(source),url=url),
            seed=args.seed,train_minimum_rows_per_character=args.train_per_character,
            validation_minimum_rows_per_character=args.validation_per_character,
            inputs={str(p):sha(p) for p in inputs},script_sha256=sha(__file__),extractor_sha256=sha(Path(__file__).with_name('build_english.py')),
            splits={s:dict(statistics(rows),quote_rows={c:sum(c in r['text'] for r in rows) for c in QUOTES},
                quote_occurrences={c:sum(r['text'].count(c) for r in rows) for c in QUOTES}) for s,rows in selected.items()},
            output_sha256={p.name:sha(p) for p in sorted(temp.iterdir())})
        (temp/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');temp.rename(args.output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print(json.dumps({s:r['quote_rows'] for s,r in report['splits'].items()},ensure_ascii=False),flush=True)
    print(args.output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=Path,default=ROOT/'charset/selected_quotes/targets.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/quote_supplement')
    p.add_argument('--seed',type=int,default=20260920)
    p.add_argument('--train-per-character',type=int,default=100)
    p.add_argument('--validation-per-character',type=int,default=25)
    build(p.parse_args())


if __name__=='__main__':main()
