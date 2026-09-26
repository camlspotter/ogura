"""Mix the Latin-sequence supplement and bind unchanged held-out sets to the new run."""
import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile
from ogura.textrec.build_english import sha
from ogura.textrec.diversity import fingerprints
from ogura.textrec.text_common import ROOT
from ogura.textrec.training.aliases import CharacterAliases


def prepare(base, supplement, suite, output, aliases_path, seed=20260922):
    if output.exists():raise FileExistsError(output)
    bm=json.loads((base/'manifest.json').read_text())
    sm=json.loads((supplement/'manifest.json').read_text())
    aliases=CharacterAliases.read(aliases_path)
    for key,name in [('training_text_sha256','train.txt'),('targets_sha256','targets.jsonl')]:
        if sha(base/name)!=bm[key]:raise ValueError(f'Base changed: {name}')
    if sm['targets_sha256']!=bm['targets_sha256'] or sm['aliases_sha256']!=sha(aliases_path):
        raise ValueError('Supplement vocabulary or aliases mismatch')
    if bm['training_text_sha256'] not in sm['excluded_files'].values():
        raise ValueError('Supplement did not exclude base training')
    for name,expected in sm['output_sha256'].items():
        if sha(supplement/name)!=expected:raise ValueError(f'Supplement changed: {name}')
    original=(base/'train.txt').read_text().splitlines()
    extra=(supplement/'train.txt').read_text().splitlines()
    rows=original+extra
    vocab={json.loads(s)['character'] for s in (base/'targets.jsonl').read_text().splitlines()}
    if len(set(rows))!=len(rows) or any(not s or not set(s)<=vocab for s in rows):
        raise ValueError('Duplicate or invalid training text')
    def fp(text):return fingerprints(text)|fingerprints(aliases.normalize(text))
    # Verify supplement diversity and its exclusion against the existing corpus.
    forbidden=set()
    for s in original:forbidden.update(fp(s))
    for s in extra:
        f=fp(s)
        if not f.isdisjoint(forbidden):raise ValueError('Supplement overlaps training text')
        forbidden.update(f)
    groups=[(name,base/name,'validation') for name in ('validation','english','validation_short5','validation_long80')]
    groups += [(name,suite/'validation'/name,'validation') for name in ('quotes','homoglyphs')]
    groups += [('test/'+name,suite/'test'/name,'test') for name in ('quotes','homoglyphs')]
    # Check new examples against held-out sets. Historical base/held-out contents
    # are preserved; do not relax their source manifest checks or promote test data.
    extra_fp=set()
    for s in extra:extra_fp.update(fp(s))
    sources={}
    for name,directory,split in groups:
        m=json.loads((directory/'manifest.json').read_text())
        expected=dict(split=split,training_text_sha256=bm['training_text_sha256'],
                      targets_sha256=bm['targets_sha256'],validation_text_sha256=sha(directory/'validation.txt'))
        if any(m.get(k)!=v for k,v in expected.items()):raise ValueError(f'Held-out manifest mismatch: {name}')
        if sha(directory/'targets.jsonl')!=bm['targets_sha256']:raise ValueError('Held-out vocabulary mismatch')
        if sha(directory/'validation.jsonl') not in sm['excluded_files'].values():
            raise ValueError(f'Supplement did not exclude held-out source: {name}')
        for s in (directory/'validation.txt').read_text().splitlines():
            if not s or not set(s)<=vocab or not fp(s).isdisjoint(extra_fp):
                raise ValueError(f'Supplement/held-out overlap or invalid text: {name}')
        sources[name]=m
    random.Random(seed).shuffle(rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.latin-mix-',dir=output.parent))
    try:
        (temp/'train.txt').write_text(''.join(s+'\n' for s in rows))
        shutil.copyfile(base/'targets.jsonl',temp/'targets.jsonl')
        training_hash=sha(temp/'train.txt')
        for name,directory,split in groups:
            dest=temp/name;dest.mkdir(parents=True)
            for filename in ('validation.txt','validation.jsonl','targets.jsonl'):
                shutil.copyfile(directory/filename,dest/filename)
            m={**sources[name],'training_text_sha256':training_hash,
               'source_directory':str(directory),'source_manifest_sha256':sha(directory/'manifest.json')}
            (dest/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
        m=dict(seed=seed,base_rows=len(original),supplement_rows=len(extra),total_rows=len(rows),
               training_text_sha256=training_hash,targets_sha256=bm['targets_sha256'],
               base_manifest_sha256=sha(base/'manifest.json'),supplement_manifest_sha256=sha(supplement/'manifest.json'),
               aliases_sha256=sha(aliases_path),script_sha256=sha(__file__),
               validation_sets=[n for n,_,s in groups if s=='validation'],reserved_test_sets=[n for n,_,s in groups if s=='test'])
        (temp/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
        temp.rename(output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    print(f'Created {output}: {len(original):,} + {len(extra):,} = {len(rows):,} rows')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'datasets/japanese_english_quotes')
    p.add_argument('--supplement',type=Path,default=ROOT/'datasets/latin_sequences_wikipedia_lowercase')
    p.add_argument('--suite',type=Path,default=ROOT/'datasets/evaluation_v2')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/japanese_english_latin_sequences')
    p.add_argument('--aliases',type=Path,default=ROOT/'config/character_aliases_quotes_homoglyphs.json')
    p.add_argument('--seed',type=int,default=20260922)
    a=p.parse_args();prepare(a.base,a.supplement,a.suite,a.output,a.aliases,a.seed)


if __name__=='__main__':main()
