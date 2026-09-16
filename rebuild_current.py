"""Rebuild the frozen current JSONL and TXT byte-for-byte from retained recipe rows."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import random
import shutil
from ogura.text_common import ROOT


def sha256(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def rebuild(recipe,output):
    spec=json.loads(recipe.read_text())
    paths=[ROOT/x['path'] for x in spec['inputs']]
    for path,entry in zip(paths,spec['inputs']):
        if sha256(path)!=entry['sha256']:raise ValueError(f'Recipe input changed: {path}')
    if output.exists():raise ValueError(f'Output already exists: {output}')
    output.mkdir(parents=True)
    offsets=array('Q')
    with (output/'train.jsonl').open('wb') as merged:
        for path in paths:
            with path.open('rb') as source:
                for line in source:
                    offsets.append(merged.tell());merged.write(line)
    assert len(offsets)==spec['samples']
    random.Random(spec['shuffle_seed']).shuffle(offsets)
    temp=output/'train.shuffled'
    with (output/'train.jsonl').open('rb') as source,temp.open('wb') as dest,(output/'train.txt').open('wb') as text:
        for offset in offsets:
            source.seek(offset);line=source.readline();dest.write(line)
            text.write((json.loads(line)['text']+'\n').encode('utf-8'))
    temp.replace(output/'train.jsonl')
    for name,expected in spec['outputs'].items():
        actual=sha256(output/name)
        if actual!=expected:raise ValueError(f'Rebuilt {name} does not match current dataset')
    for entry in spec['metadata']:
        path=ROOT/entry['path']
        if sha256(path)!=entry['sha256']:raise ValueError(f'Metadata changed: {path}')
        shutil.copyfile(path,output/path.name)
    (output/'rebuild_verification.json').write_text(json.dumps({'samples':spec['samples'],'byte_identical':True,'sha256':spec['outputs']},indent=2)+'\n')
    print(f'Byte-identical rebuild verified: {output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=ROOT/'current_dataset.json')
    args=p.parse_args();rebuild(args.recipe,args.output)


if __name__=='__main__':main()
