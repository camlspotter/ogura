"""Deterministically shuffle JSONL rows before final verification/plaintext export."""
import argparse
from array import array
import json
from pathlib import Path
import random


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260915)
    args=p.parse_args();path=args.output/'train.jsonl'
    marker=args.output/'shuffle.json'
    if marker.exists():
        assert json.loads(marker.read_text())['seed']==args.seed
        return
    offsets=array('Q')
    with path.open('rb') as f:
        while True:
            offset=f.tell()
            if not f.readline():break
            offsets.append(offset)
        random.Random(args.seed).shuffle(offsets)
        temp=path.with_suffix('.shuffling')
        with temp.open('wb') as out:
            for offset in offsets:f.seek(offset);out.write(f.readline())
    temp.replace(path)
    marker.write_text(json.dumps({'seed':args.seed,'samples':len(offsets),'algorithm':'seeded Fisher-Yates via Python random.shuffle'})+'\n')
    for name in ('manifest.json','summary.json'):
        p=args.output/name;r=json.loads(p.read_text());r['shuffle_seed']=args.seed;p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
