"""Build targeted Latin-sequence validation without touching reserved test articles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import pyarrow.parquet as pq
from ogura.build_english import download, sha, CONFIG, REPO, REVISION, SHARD, SHA256
from ogura.collect_latin_sequences import PATTERNS, paragraphs, excerpts, training_article
from ogura.diversity import fingerprints
from ogura.text_common import ROOT
from ogura.training.aliases import CharacterAliases


def eligible(article_id):
    # Identical reserved-test partition to build_evaluation_suite; never evaluate it.
    digest=hashlib.sha256(f'evaluation-test:{article_id}'.encode()).hexdigest()
    return not training_article(article_id) and int(digest[:8],16)%5 != 0


def build(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.goal<1:raise ValueError('goal must be positive')
    source,url=download(args.corpus)
    aliases=CharacterAliases.read(args.aliases)
    targets=args.base/'targets.jsonl';train=args.base/'train.txt'
    bm=json.loads((args.base/'manifest.json').read_text())
    if sha(train)!=bm['training_text_sha256'] or sha(targets)!=bm['targets_sha256']:
        raise ValueError('Training artifacts differ from manifest')
    vocabulary={json.loads(s)['character'] for s in targets.read_text().splitlines()}
    def fp(text):return fingerprints(text)|fingerprints(aliases.normalize(text))
    forbidden=set(); excluded_articles=set(); inputs={str(train):sha(train),str(targets):sha(targets)}
    for line in train.read_text().splitlines():forbidden.update(fp(line))
    for path in sorted(set(args.exclude_jsonl)):
        inputs[str(path)]=sha(path)
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('language')=='en' or '//en.wikipedia.org/' in row.get('url',''):
                excluded_articles.add(str(row['article_id']))
            forbidden.update(fp(row['text']))
    rows=[]; counts=Counter(); scanned=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256,columns=['id','url','title','text']):
        for article in batch.to_pylist():
            scanned+=1;aid=str(article['id'])
            if not eligible(aid) or aid in excluded_articles:continue
            used=set()
            for offset,paragraph in paragraphs(article):
                needed={p for p in PATTERNS if counts[p]<args.goal and p not in used}
                if not needed or not any(p in paragraph for p in needed):continue
                for row in excerpts(paragraph,offset,vocabulary):
                    if not any(counts[p]<args.goal and p not in used for p in row['patterns']):continue
                    f=fp(row['text'])
                    if not f.isdisjoint(forbidden):continue
                    assert article['text'][row['start']:row['end']]==row['text']
                    row.update(article_id=aid,title=article['title'],url=article['url'],language='en',split='validation',
                               sample_id=hashlib.sha256(row['text'].encode()).hexdigest())
                    rows.append(row);counts.update(row['patterns']);used.update(row['patterns']);forbidden.update(f)
        if scanned%25600==0:print(f'Scanned {scanned:,}; selected {len(rows):,}',flush=True)
    if not rows:raise ValueError('No validation candidates')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.latin-validation-',dir=args.output.parent))
    try:
        (temp/'validation.txt').write_text(''.join(r['text']+'\n' for r in rows))
        (temp/'validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        shutil.copyfile(targets,temp/'targets.jsonl')
        lengths=[len(r['text']) for r in rows]
        manifest=dict(split='validation',samples=len(rows),min_length=min(lengths),max_length=max(lengths),
                      training_text_sha256=sha(train),targets_sha256=sha(targets),validation_text_sha256=sha(temp/'validation.txt'),
                      aliases_sha256=sha(args.aliases),goal=args.goal,scanned_articles=scanned,
                      patterns={p:dict(rows=counts[p],shortfall=max(0,args.goal-counts[p])) for p in PATTERNS},
                      source=dict(repository=REPO,revision=REVISION,config=CONFIG,shard=SHARD,sha256=SHA256,url=url),
                      excluded_files=inputs,script_sha256=sha(__file__),
                      dependencies_sha256={str(p):sha(p) for p in [ROOT/'ogura/collect_latin_sequences.py',ROOT/'ogura/build_english.py',ROOT/'ogura/diversity.py',ROOT/'ogura/training/aliases.py']},
                      output_sha256={p.name:sha(p) for p in temp.iterdir()},
                      notes=['Lowercase patterns; overlapping sequences share examples.',
                             'Validation-only article partition; reserved test and known source articles excluded.',
                             'Vocabulary words may overlap training; source articles and 16-character fragments do not.'])
        (temp/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        temp.rename(args.output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print(f'Created {args.output}: {len(rows):,} rows; {dict(counts)}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'datasets/japanese_english_latin_sequences')
    p.add_argument('--corpus',type=Path,default=ROOT/'corpus/wikipedia'/CONFIG)
    p.add_argument('--output',type=Path,default=ROOT/'datasets/validation_latin_sequences')
    p.add_argument('--aliases',type=Path,default=ROOT/'config/character_aliases_quotes_homoglyphs.json')
    p.add_argument('--goal',type=int,default=200)
    p.add_argument('--exclude-jsonl',type=Path,action='append')
    args=p.parse_args()
    if args.exclude_jsonl is None:
        args.exclude_jsonl=list((ROOT/'datasets').rglob('validation.jsonl'))+list((ROOT/'datasets').rglob('test.jsonl'))
        args.exclude_jsonl += [ROOT/'datasets'/name/'train.jsonl' for name in ('english_wikipedia_30k','quote_supplement','latin_sequences_wikipedia_lowercase')]
        probe=ROOT/'runs/noto48-residual64-quotes-homoglyphs/comparison/probe/validation.jsonl'
        if probe.exists():args.exclude_jsonl.append(probe)
    build(args)


if __name__=='__main__':main()
