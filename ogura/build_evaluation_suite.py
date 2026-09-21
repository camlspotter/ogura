"""Build reproducible, disjoint quote/homoglyph validation and reserved test sets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import unicodedata

import pyarrow.parquet as pq
from ogura.build_english import candidate as english_candidate, download, CONFIG, REVISION
from ogura.build_validation import sha256
from ogura.diversity import fingerprints
from ogura.prose_filter import is_prose
from ogura.text_common import ROOT, split_article, strip_han_ivs, digest
from ogura.training.aliases import CharacterAliases

GROUPS = {'quotes': '“”‘’', 'homoglyphs': 'ËЁëёΠПΦФΓГ'}


def candidates(article, vocabulary, characters, language):
    """Literal excerpts; require prose in the parent paragraph, reject code/tables."""
    if language == 'en':
        for c in characters:
            if c in article['text']:
                row = english_candidate(article, vocabulary, 20260920, required_character=c)
                if row:
                    yield row
        return
    for line in re.finditer(r'[^\n]+', article['text']):
        text = line.group()
        if not set(text) & set(characters) or not is_prose(text, article['title']):
            continue
        if re.search(r'https?://|[{}|\\]|\'\'\'|:en:', text):
            continue
        for hit in re.finditer('[' + re.escape(characters) + ']', text):
            for shift in (12, 18, 6, 0):
                start = max(0, min(hit.start()-shift, len(text)-25))
                end = min(start+25, len(text))
                raw = text[start:end]
                value = strip_han_ivs(raw)
                if (20 <= len(value) <= 25 and hit.group() in value and set(value) <= vocabulary
                        and not value[0].isspace() and not value[-1].isspace()
                        and not unicodedata.category(value[0]).startswith('M')):
                    yield dict(text=value, article_id=article['id'], title=article['title'], url=article['url'],
                               start=line.start()+start, end=line.start()+end)


def choose(pool, characters, goal, forbidden, aliases, used_articles, seed):
    """Balance rare source characters; shortages are reported, never synthesized."""
    pool = list(pool)
    random.Random(seed).shuffle(pool)
    counts = Counter()
    chosen = []
    exhausted = set()
    while True:
        todo = [c for c in characters if counts[c] < goal and c not in exhausted]
        if not todo:
            return chosen, dict(counts)
        c = min(todo, key=lambda c: (sum(c in r['text'] for r in pool), characters.index(c)))
        found = False
        for row in pool:
            key = (row['language'], str(row['article_id']))
            if c not in row['text'] or key in used_articles:
                continue
            fp = fingerprints(row['text']) | fingerprints(aliases.normalize(row['text']))
            if not fp.isdisjoint(forbidden):
                continue
            chosen.append(row); used_articles.add(key); forbidden.update(fp)
            counts.update(sorted(set(row['text']) & set(characters)))
            found = True
            if counts[c] >= goal:
                break
        if not found:
            exhausted.add(c)


def build(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.quote_goal, args.homoglyph_goal, args.test_goal) < 1:
        raise ValueError('Goals must be positive')
    base = args.base
    aliases = CharacterAliases.read(args.aliases)
    targets = base/'targets.jsonl'
    vocabulary = {json.loads(l)['character'] for l in targets.read_text().splitlines()}
    if not set(''.join(GROUPS.values())) <= vocabulary:
        raise ValueError('Required characters missing from vocabulary')
    forbidden = set(); inputs = {targets, args.aliases}; used_en = set()
    # Exclude all prior English training articles; keep already-used validation out of reserved test.
    prior_validation = set()
    for lang, paths in [('en', [ROOT/'datasets/english_wikipedia_30k/train.jsonl', ROOT/'datasets/quote_supplement/train.jsonl']),
                        ('validation', list(base.glob('*/validation.jsonl')))]:
        for path in paths:
            inputs.add(path)
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if lang == 'en': used_en.add(str(row['article_id']))
                else: prior_validation.add(str(row['article_id']))
    for path in [base/'train.txt', *base.glob('*/validation.txt')]:
        inputs.add(path)
        for text in path.read_text().splitlines():
            forbidden.update(fingerprints(text)); forbidden.update(fingerprints(aliases.normalize(text)))
    # The previous diagnostic probe is already used for model selection, so cannot become test data.
    prior_probe_fragments = set()
    if args.prior_probe and args.prior_probe.exists():
        inputs.add(args.prior_probe)
        for line in args.prior_probe.read_text().splitlines():
            row = json.loads(line)
            prior_validation.add(str(row['article_id']))
            prior_probe_fragments.update(fingerprints(row['text']))
            prior_probe_fragments.update(fingerprints(aliases.normalize(row['text'])))
    english, _ = download(ROOT/'corpus/wikipedia'/CONFIG)
    japanese = sorted((ROOT/'corpus/wikipedia/20231101.ja').glob('*.parquet'))
    if not japanese:
        raise FileNotFoundError('Japanese Wikipedia parquet files are required')
    pools = {(split, group): [] for split in ('validation','test') for group in GROUPS}
    sources = [('ja', p) for p in japanese] + [('en', english)]
    for language, path in sources:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['id','url','title','text']):
            for article in batch.to_pylist():
                aid = str(article['id'])
                if language == 'ja':
                    partition = split_article(aid, 20260915)
                    if partition == 'train': continue
                else:
                    if aid in used_en: continue
                    if int(hashlib.sha256(f'20260920:split:{aid}'.encode()).hexdigest(),16)%10 != 0: continue
                    partition = 'test' if int(digest(f'evaluation-test:{aid}')[:8],16)%5 == 0 else 'validation'
                if partition == 'test' and aid in prior_validation: continue
                for group, chars in GROUPS.items():
                    if not set(chars) & set(article['text']): continue
                    seen = set()
                    for row in candidates(article, vocabulary, chars, language):
                        if row['text'] in seen: continue
                        seen.add(row['text'])
                        fp = fingerprints(row['text']) | fingerprints(aliases.normalize(row['text']))
                        if not fp.isdisjoint(forbidden): continue
                        if partition == 'test' and not fp.isdisjoint(prior_probe_fragments): continue
                        row.update(language=language, split=partition, sample_id=digest(row['text']))
                        pools[partition,group].append(row)
        print(f'Scanned {path.name} ({language})', flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.evaluation-', dir=args.output.parent))
    summaries = {}
    prior_quotes = [dict(json.loads(line), language='en', split='validation')
                    for line in (base/'quotes/validation.jsonl').read_text().splitlines()]
    used = {('en', str(row['article_id'])) for row in prior_quotes}
    try:
        # Reserve test first; it is never passed to training or scored by this command.
        for split in ('test','validation'):
            for group, chars in GROUPS.items():
                goal = args.test_goal if split == 'test' else (args.quote_goal if group == 'quotes' else args.homoglyph_goal)
                rows, counts = choose(pools[split,group], chars, goal, forbidden, aliases, used, args.seed)
                if split == 'validation' and group == 'quotes':
                    rows = prior_quotes + rows
                    counts = dict(Counter(c for row in rows for c in sorted(set(row['text']) & set(chars))))
                if not rows: raise ValueError(f'No eligible {split}/{group} samples')
                dest = temp/split/group; dest.mkdir(parents=True)
                (dest/'validation.txt').write_text(''.join(r['text']+'\n' for r in rows))
                (dest/'validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
                shutil.copyfile(targets,dest/'targets.jsonl')
                manifest = dict(samples=len(rows),min_length=20,max_length=25,split=split,
                    training_text_sha256=sha256(base/'train.txt'),targets_sha256=sha256(targets),
                    validation_text_sha256=sha256(dest/'validation.txt'),source_character_rows=counts,
                    requested_rows_per_character=goal,shortfalls={c:goal-counts.get(c,0) for c in chars if counts.get(c,0)<goal},
                    one_sample_per_article=True,no_shared_raw_or_normalized_16_character_fragments=True)
                (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
                summaries[f'{split}/{group}'] = manifest
        report = dict(seed=args.seed,revision=REVISION,inputs={str(p):sha256(p) for p in sorted(inputs)},
            corpus={str(p):sha256(p) for _,p in sources},script_sha256=sha256(Path(__file__)),
            dependencies_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('build_english.py','prose_filter.py','diversity.py','text_common.py')},
            aliases=aliases.config,splits=summaries)
        (temp/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temp.rename(args.output)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    for name, info in summaries.items(): print(name,info['samples'],info['source_character_rows'],'shortfalls',info['shortfalls'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'datasets/japanese_english_quotes')
    p.add_argument('--aliases',type=Path,default=ROOT/'config/character_aliases_quotes_homoglyphs.json')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/evaluation_v2')
    p.add_argument('--prior-probe',type=Path,default=ROOT/'runs/noto48-residual64-quotes-homoglyphs/comparison/probe/validation.jsonl')
    p.add_argument('--quote-goal',type=int,default=100)
    p.add_argument('--homoglyph-goal',type=int,default=50)
    p.add_argument('--test-goal',type=int,default=20)
    p.add_argument('--seed',type=int,default=20260922)
    build(p.parse_args())


if __name__=='__main__':main()
