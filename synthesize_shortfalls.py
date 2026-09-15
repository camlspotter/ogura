"""Fill deficits by same-class substitution in randomly sampled corpus excerpts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import unicodedata as ud
from extract_training_text import ROOT, digest, emit


def char_class(c):
    if ud.name(c,'').startswith(('CJK UNIFIED IDEOGRAPH-','CJK COMPATIBILITY IDEOGRAPH-')):return 'han'
    return ud.category(c)


def apply_replacements(base, replacements):
    chars=list(base);seen=set()
    for edit in replacements:
        i=edit['position'];before=edit['from'];after=edit['to']
        if i in seen or not 0<=i<len(chars) or chars[i]!=before or len(after)!=1:
            raise ValueError('Invalid replacement')
        if char_class(before)!=char_class(after):raise ValueError('Replacement class mismatch')
        seen.add(i);chars[i]=after
    return ''.join(chars)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'data/training_text_direct_100')
    p.add_argument('--output',type=Path,default=ROOT/'data/training_text_final_100')
    p.add_argument('--report-dir',type=Path,default=ROOT/'results/training_text_final_100')
    p.add_argument('--goal',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260915)
    args=p.parse_args()
    if args.goal<1:p.error('goal must be positive')
    out=args.output;report=args.report_dir
    if (out/'train.jsonl').exists():p.error('Output exists; choose a fresh directory')
    targets=list(map(json.loads,(args.base/'targets.jsonl').open()))
    vocab={r['character'] for r in targets}
    original=json.loads((args.base/'summary.json').read_text())
    min_length=original.get('min_tokens',1)
    coverage=list(map(json.loads,(args.base/'coverage.jsonl').open()))
    needed={r['character']:max(0,args.goal-r['sample_count']) for r in coverage}
    classes={char_class(c) for c,n in needed.items() if n}
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    pools=defaultdict(list);counts=Counter();lengths=Counter();hashes=set()
    with (args.base/'train.jsonl').open('rb') as source,(out/'train.jsonl').open('w') as merged:
        while True:
            offset=source.tell();line=source.readline()
            if not line:break
            row=json.loads(line);text=row['text']
            assert set(text)<=vocab and min_length<=len(text)<=25
            counts.update(set(text));lengths[len(text)]+=1;hashes.add(row['sample_id']);emit(merged,row)
            for cls in {char_class(c) for c in text}&classes:pools[cls].append(offset)
    for c,n in needed.items():
        if n and not pools[char_class(c)]:raise ValueError(f'No donor excerpts for {c!r}')
    rng=random.Random(args.seed);added=0
    with (args.base/'train.jsonl').open('rb') as source,(out/'train.jsonl').open('a') as merged,(out/'synthetic.jsonl').open('w') as synthetic:
        for c in sorted(needed,key=lambda c:(-needed[c],c)):
            attempts=0
            while counts[c]<args.goal:
                attempts+=1
                if attempts>max(10000,args.goal*1000):raise RuntimeError(f'Could not fill {c!r}; choose another seed or donor corpus')
                source.seek(rng.choice(pools[char_class(c)]));base=json.loads(source.readline());text=base['text']
                # The new target must be newly introduced, rather than an incidental repeat.
                if c in text:continue
                positions=[i for i,x in enumerate(text) if char_class(x)==char_class(c)]
                i=rng.choice(positions);edits=[{'position':i,'from':text[i],'to':c}]
                modified=apply_replacements(text,edits);key=digest(modified)
                if key in hashes:continue
                row={**base,'sample_id':key,'text':modified,'anchor':c,'extraction':'synthetic_substitution',
                     'synthetic':True,'base_sample_id':base['sample_id'],'base_text':text,'replacements':edits}
                emit(merged,row);emit(synthetic,row);hashes.add(key);counts.update(set(modified));lengths[len(modified)]+=1;added+=1
    assert all(counts[c]>=args.goal for c in vocab)
    for name in ('targets.jsonl','article_splits.jsonl'):shutil.copyfile(args.base/name,out/name)
    metadata=json.loads((args.base/'manifest.json').read_text())
    metadata.update({'target_samples_per_entry':args.goal,'synthesis_seed':args.seed,'synthesis_base':str(args.base.resolve()),
        'synthesis':'uniform random donor excerpt from natural selected training-corpus excerpts with matching character class; uniform matching position; one replacement',
        'synthesis_class':'Han ideographs replace Han; other characters match Unicode general category (JIS symbol: So)',
        'limitations':'Synthetic strings may be linguistically unnatural; 100 is an initial collection target, not an accuracy guarantee.'})
    with (out/'coverage.jsonl').open('w') as f:
        for r in targets:
            c=r['character'];emit(f,{'character':c,'sample_count':counts[c],'shortfall':0,'status':'met','corpus_occurrences':r['ranking_occurrences']})
    summary={**metadata,'articles':original['articles'],'article_splits':original['article_splits'],'samples':len(hashes),
             'natural_samples':original['samples'],'synthetic_samples':added,'length_counts':dict(sorted(lengths.items())),
             'coverage_status':{'met':len(vocab)}}
    for name,obj in [('manifest.json',metadata),('summary.json',summary)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    for name in ('manifest.json','summary.json','coverage.jsonl'):shutil.copyfile(out/name,report/name)
    (report/'shortfalls.jsonl').write_text('')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
