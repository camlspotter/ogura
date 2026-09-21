"""Add the quote supplement without changing existing training/validation artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import tempfile
from ogura.analyze_wikipedia import ROOT
from ogura.build_english import sha
from ogura.diversity import fingerprints


def prepare(base, supplement, output, seed=20260921):
    if output.exists():raise FileExistsError(output)
    bm=json.loads((base/'manifest.json').read_text())
    sm=json.loads((supplement/'manifest.json').read_text())
    for key,name in [('training_text_sha256','train.txt'),('targets_sha256','targets.jsonl')]:
        if sha(base/name)!=bm[key]:raise ValueError(f'Base changed: {name}')
    for name,expected in sm['output_sha256'].items():
        if sha(supplement/name)!=expected:raise ValueError(f'Supplement changed: {name}')
    if sha(base/'train.txt') not in sm['inputs'].values():raise ValueError('Supplement was not checked against base training')
    targets=supplement/'targets.jsonl'
    vocab={json.loads(s)['character'] for s in targets.read_text().splitlines()}
    old={json.loads(s)['character'] for s in (base/'targets.jsonl').read_text().splitlines()}
    if vocab-old!=set('“”‘’') or old-vocab:raise ValueError('Expected exactly four added quotes')
    original=(base/'train.txt').read_text().splitlines()
    extra=(supplement/'train.txt').read_text().splitlines()
    rows=original+extra
    if len(set(rows))!=len(rows):raise ValueError('Duplicate training text')
    if any(not s or not set(s)<=vocab for s in rows):raise ValueError('Invalid training labels')
    forbidden=set()
    for s in rows:forbidden.update(fingerprints(s))
    groups=[(name,base/name/'validation.txt') for name in bm['validation_sets']]
    groups.append(('quotes',supplement/'validation.txt'))
    for name,path in groups:
        if name!='quotes':
            m=json.loads((path.parent/'manifest.json').read_text())
            expected=dict(training_text_sha256=bm['training_text_sha256'],targets_sha256=bm['targets_sha256'],validation_text_sha256=sha(path),split='validation')
            if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Invalid base validation manifest')
        for s in path.read_text().splitlines():
            if not s or not set(s)<=vocab:raise ValueError('Invalid validation labels')
            if not forbidden.isdisjoint(fingerprints(s)):raise ValueError(f'Validation/train overlap: {name}')
    random.Random(seed).shuffle(rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.quote-mix-',dir=output.parent))
    try:
        (temp/'train.txt').write_text(''.join(s+'\n' for s in rows))
        shutil.copyfile(targets,temp/'targets.jsonl')
        for name,path in groups:
            dest=temp/name;dest.mkdir()
            shutil.copyfile(path,dest/'validation.txt');shutil.copyfile(targets,dest/'targets.jsonl')
            if path.with_suffix('.jsonl').exists():shutil.copyfile(path.with_suffix('.jsonl'),dest/'validation.jsonl')
            lengths=[len(s) for s in path.read_text().splitlines()]
            m=dict(split='validation',samples=len(lengths),min_length=min(lengths),max_length=max(lengths),
                training_text_sha256=sha(temp/'train.txt'),targets_sha256=sha(targets),validation_text_sha256=sha(path),
                source_manifest_sha256=sha(path.parent/'manifest.json'))
            (dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        counts=Counter(''.join(rows))
        m=dict(seed=seed,base_rows=len(original),quote_rows=len(extra),total_rows=len(rows),
               quote_occurrences={c:counts[c] for c in '“”‘’'},training_text_sha256=sha(temp/'train.txt'),
               targets_sha256=sha(targets),base_manifest_sha256=sha(base/'manifest.json'),
               supplement_manifest_sha256=sha(supplement/'manifest.json'),script_sha256=sha(__file__),
               validation_sets=[n for n,_ in groups])
        (temp/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
        temp.rename(output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print(f'Created {output}: {len(original):,} + {len(extra):,} = {len(rows):,} training rows')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'datasets/japanese_english_30k')
    p.add_argument('--supplement',type=Path,default=ROOT/'datasets/quote_supplement')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/japanese_english_quotes')
    p.add_argument('--seed',type=int,default=20260921)
    a=p.parse_args();prepare(a.base,a.supplement,a.output,a.seed)


if __name__=='__main__':main()
