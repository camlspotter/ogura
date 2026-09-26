"""Collect targeted word sequences from pinned English Wikipedia training articles."""
from ogura.textrec.paths import CORPUS_ROOT

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import pyarrow.parquet as pq
from ogura.textrec.build_english import CONFIG, SHARD, SHA256, REVISION, REPO, download, prose, sha, LATIN
from ogura.textrec.diversity import fingerprints
from ogura.textrec.text_common import ROOT
from ogura.textrec.training.aliases import CharacterAliases

PATTERNS = ('ff','ll','tt','ii','fi','fl','ffi','ffl','il','li','ij','ji','ft','ti','it','lt','tl')
WORD = re.compile(r"(?<![\w'-])[A-Za-z]+(?:['-][A-Za-z]+)*(?![\w'-])")
TITLE = re.compile(r'^(?:List of|Lists of|Index of|Outline of|Glossary of)\b|alphabet|Unicode|disambiguation', re.I)


def training_article(article_id):
    # Preserve the original English corpus split, including reserved evaluation-v2 test.
    return int(hashlib.sha256(f'20260920:split:{article_id}'.encode()).hexdigest(),16)%10 != 0


def matched_words(text):
    return [m for m in WORD.finditer(text)
            if any(p in m.group() for p in PATTERNS)]


def paragraphs(article):
    if TITLE.search(article['title']): return
    for m in re.finditer(r'[^\n]+', article['text']):
        if prose(m.group()): yield m.start(), m.group()


def excerpts(paragraph, offset, vocabulary):
    """Literal 20–25 character, whole-token windows containing targeted words."""
    tokens = list(re.finditer(r'\S+', paragraph))
    for i, first in enumerate(tokens):
        for last in tokens[i+1:]:
            text = paragraph[first.start():last.end()]
            if len(text)>25: break
            if len(text)<20 or not set(text)<=vocabulary: continue
            if len(LATIN.findall(text))/len(text)<.7: continue
            if not all(c==' ' or (ord(c)<=255 and (c.isalpha() or c.isdigit())) or c in ",.;:!?'-\"()“”‘’–—" for c in text):continue
            words = [m.group() for m in matched_words(text)]
            if not words: continue
            yield dict(text=text,start=offset+first.start(),end=offset+last.end(),
                       words=words,patterns=[p for p in PATTERNS if any(p in w for w in words)])


def build(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.goal<1:raise ValueError('goal must be positive')
    source,url=download(args.corpus)
    aliases=CharacterAliases.read(args.aliases)
    vocab={json.loads(s)['character'] for s in args.targets.read_text().splitlines()}
    exclusions=sorted(set(args.exclude_jsonl))
    forbidden=set(); heldout=set(); inputs={}
    for path in exclusions:
        inputs[str(path)]=sha(path)
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if row.get('language')=='en' or '//en.wikipedia.org/' in row.get('url',''):
                heldout.add(str(row['article_id']))
            for text in (row['text'],aliases.normalize(row['text'])):forbidden.update(fingerprints(text))
    # Keep supplements distinct from existing training lines, including aliases.
    inputs[str(args.training_text)]=sha(args.training_text)
    for line in args.training_text.read_text().splitlines():
        forbidden.update(fingerprints(line));forbidden.update(fingerprints(aliases.normalize(line)))
    rows=[]; counts=Counter(); words={};scanned=0;eligible=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256,columns=['id','url','title','text']):
        for article in batch.to_pylist():
            scanned+=1;aid=str(article['id'])
            if not training_article(aid) or aid in heldout:continue
            eligible+=1; used_patterns=set()
            for offset,paragraph in paragraphs(article):
                matches=matched_words(paragraph)
                for m in matches:
                    word=m.group(); key=word
                    if key not in words:
                        words[key]=dict(word=key,occurrences=0,patterns=[p for p in PATTERNS if p in key],
                            source=dict(article_id=aid,title=article['title'],url=article['url'],
                                        start=offset+m.start(),end=offset+m.end(),text=word))
                    words[key]['occurrences']+=1
                needed={p for p in PATTERNS if counts[p]<args.goal and p not in used_patterns}
                if not needed or not any(p in m.group() for m in matches for p in needed):continue
                for row in excerpts(paragraph,offset,vocab):
                    if not any(counts[p]<args.goal and p not in used_patterns for p in row['patterns']):continue
                    fp=fingerprints(row['text'])|fingerprints(aliases.normalize(row['text']))
                    if not fp.isdisjoint(forbidden):continue
                    assert article['text'][row['start']:row['end']]==row['text']
                    row.update(article_id=aid,title=article['title'],url=article['url'],language='en',split='train',
                               sample_id=hashlib.sha256(row['text'].encode()).hexdigest())
                    rows.append(row);forbidden.update(fp);counts.update(row['patterns']);used_patterns.update(row['patterns'])
        if scanned%10240==0:print(f'Scanned {scanned:,}; selected {len(rows):,}',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.latin-sequences-',dir=args.output.parent))
    try:
        def jsonl(name, items):
            (temp/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
        jsonl('train.jsonl',rows)
        (temp/'train.txt').write_text(''.join(r['text']+'\n' for r in rows))
        jsonl('words.jsonl',[words[w] for w in sorted(words)])
        report=dict(source=dict(repository=REPO,revision=REVISION,config=CONFIG,shard=SHARD,sha256=SHA256,url=url,
                               license='Wikipedia CC BY-SA / GFDL; retain article attribution'),
                    scanned_articles=scanned,eligible_training_articles=eligible,rows=len(rows),goal=args.goal,
                    patterns={p:dict(selected_rows=counts[p],shortfall=max(0,args.goal-counts[p]),
                                     word_types=sum(p in w for w in words),
                                     corpus_word_occurrences=sum(v['occurrences'] for w,v in words.items() if p in w)) for p in PATTERNS},
                    notes=['Case-sensitive lowercase sequence search, original text preserved; uppercase Roman numerals do not count as ii.',
                           'Words are tokens from English Wikipedia prose; names and foreign words are not dictionary-filtered.',
                           'Overlapping patterns share rows, not duplicated training entries.',
                           'Word inventory covers eligible prose; excerpts additionally enforce vocabulary, length and fragment exclusions.',
                           'Fixed source order; no synthesis or model/training mixture changes.'],
                    excluded_files=inputs,targets_sha256=sha(args.targets),aliases_sha256=sha(args.aliases),
                    implementation_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('build_english.py'),Path(__file__).with_name('diversity.py'),ROOT/'ogura/textrec/training/aliases.py']},
                    output_sha256={p.name:sha(p) for p in temp.iterdir()})
        (temp/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temp.rename(args.output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print(json.dumps(report['patterns'],ensure_ascii=False,indent=2));print(f'Created {args.output}: {len(rows):,} rows, {len(words):,} word types')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=CORPUS_ROOT / 'wikipedia'/CONFIG)
    p.add_argument('--output',type=Path,default=ROOT/'datasets/latin_sequences_wikipedia')
    p.add_argument('--targets',type=Path,default=ROOT/'datasets/japanese_english_quotes/targets.jsonl')
    p.add_argument('--training-text',type=Path,default=ROOT/'datasets/japanese_english_quotes/train.txt')
    p.add_argument('--aliases',type=Path,default=ROOT/'config/character_aliases_quotes_homoglyphs.json')
    p.add_argument('--exclude-jsonl',type=Path,action='append')
    p.add_argument('--goal',type=int,default=1000)
    args=p.parse_args()
    if args.exclude_jsonl is None:
        args.exclude_jsonl=sorted((ROOT/'datasets').rglob('validation.jsonl'))
        args.exclude_jsonl+=sorted((ROOT/'datasets').rglob('test.jsonl'))
        probe=ROOT/'runs/noto48-residual64-quotes-homoglyphs/comparison/probe/validation.jsonl'
        if probe.exists():args.exclude_jsonl.append(probe)
    build(args)


if __name__=='__main__':main()
