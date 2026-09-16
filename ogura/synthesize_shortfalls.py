"""Fill deficits by same-class substitution in randomly sampled corpus excerpts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import sqlite3
import zlib
import pickle
import unicodedata as ud
from ogura.text_common import ROOT, digest, emit, LINE, TOKEN, strip_han_ivs
from ogura.diversity import DiversityFilter, VERSION, SHINGLE_LENGTH
from ogura.prose_filter import is_prose


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
    p.add_argument('--base',type=Path,default=ROOT/'datasets/direct_100_len20_25_diverse')
    p.add_argument('--output',type=Path,default=ROOT/'datasets/final_100_len20_25_diverse')
    p.add_argument('--report-dir',type=Path,default=ROOT/'datasets/final_100_len20_25_diverse/reports')
    p.add_argument('--goal',type=int,default=100)
    p.add_argument('--donor-index',type=Path,default=ROOT/'cache/wikipedia_source_index')
    p.add_argument('--seed',type=int,default=20260915)
    p.add_argument('--resume',action='store_true',help='Resume incomplete unshuffled output; record prefix hash as RNG checkpoint')
    args=p.parse_args()
    if args.goal<1:p.error('goal must be positive')
    out=args.output;report=args.report_dir
    if (out/'train.jsonl').exists() and not args.resume:p.error('Output exists; choose a fresh directory or --resume')
    if args.resume and ((out/'summary.json').exists() or (out/'shuffle.json').exists()):p.error('Cannot resume completed or shuffled output')
    targets=list(map(json.loads,(args.base/'targets.jsonl').open()))
    vocab={r['character'] for r in targets}
    original=json.loads((args.base/'summary.json').read_text())
    min_length=original.get('min_tokens',1)
    coverage=list(map(json.loads,(args.base/'coverage.jsonl').open()))
    needed={r['character']:max(0,args.goal-r['sample_count']) for r in coverage}
    classes={char_class(c) for c,n in needed.items() if n}
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    counts=Counter();lengths=Counter();hashes=set();diversity=DiversityFilter();donor_hashes=set();added=0
    resume_info={}
    if args.resume:
        import hashlib
        with (out/'train.jsonl').open('rb') as stream:prefix_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        with (out/'train.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line);text=row['text'];donor=row.get('base_text',text)
                assert set(text)<=vocab and min_length<=len(text)<=25
                assert diversity.allows(row['article_id'],row['start'],row['end'],donor)
                diversity.add(row['article_id'],row['start'],row['end'],donor)
                counts.update(set(text));lengths[len(text)]+=1;hashes.add(row['sample_id'])
                if row.get('synthetic'):
                    donor_hashes.add(digest(donor));added+=1
        resume_info={'resume_prefix_sha256':prefix_hash,'resumed_synthetic_samples':added}
    else:
        with (args.base/'train.jsonl').open() as source,(out/'train.jsonl').open('w') as merged:
            for line in source:
                row=json.loads(line);text=row['text']
                assert set(text)<=vocab and min_length<=len(text)<=25
                assert diversity.allows(row['article_id'],row['start'],row['end'],text)
                diversity.add(row['article_id'],row['start'],row['end'],text)
                counts.update(set(text));lengths[len(text)]+=1;hashes.add(row['sample_id']);emit(merged,row)
    db=sqlite3.connect(f"file:{args.donor_index/'articles.sqlite3'}?mode=ro",uri=True)
    with (args.donor_index/'complete.pkl').open('rb') as f:
        inv,_,_,article_total,index_chars,revision,split_seed=pickle.load(f)
    assert revision==original['source_revision'] and split_seed==original['seed']
    rng=random.Random(f'{args.seed}:{resume_info["resume_prefix_sha256"]}' if args.resume else args.seed)
    with (out/'train.jsonl').open('a') as merged,(out/'synthetic.jsonl').open('a' if args.resume else 'w') as synthetic:
        for c in sorted(needed,key=lambda c:(-needed[c],c)):
            attempts=0;cls=char_class(c)
            while counts[c]<args.goal:
                attempts+=1
                if attempts>max(100000,args.goal*10000):raise RuntimeError(f'Could not fill {c!r}; insufficient distinct donors')
                n=rng.randrange(article_total)
                result=db.execute('SELECT payload FROM articles WHERE n=?',(n,)).fetchone()
                if result is None:continue
                article=json.loads(zlib.decompress(result[0]));raw=article['text']
                if len(raw)<min_length:continue
                length=rng.randint(min_length,25)
                if cls=='han':
                    start=rng.randrange(max(1,len(raw)-min_length+1))
                else:
                    positions=[i for i,x in enumerate(raw) if char_class(x)==cls]
                    if not positions:continue
                    at=rng.choice(positions);start=max(0,at-rng.randrange(length))
                line=LINE.match(raw,start)
                if line is None:continue
                ms=list(TOKEN.finditer(raw[start:min(line.end(),start+60)]))[:length]
                if len(ms)<min_length:continue
                text=strip_han_ivs(''.join(m.group() for m in ms));finish=start+ms[-1].end()
                if not min_length<=len(text)<=25:continue
                if text[0].isspace() or text[-1].isspace() or ud.category(text[0]).startswith('M'):continue
                if finish<len(raw) and ud.category(raw[finish]).startswith('M'):continue
                if not is_prose(text,article['title']):continue
                if digest(text) in hashes:continue
                if c in text or not diversity.allows(article['id'],start,finish,text):continue
                unknown=[i for i,x in enumerate(text) if x not in vocab]
                if len(unknown)>1:continue
                positions=[i for i,x in enumerate(text) if char_class(x)==cls and (not unknown or i==unknown[0])]
                if not positions:continue
                i=rng.choice(positions);edits=[{'position':i,'from':text[i],'to':c}]
                modified=apply_replacements(text,edits);key=digest(modified)
                assert set(modified)<=vocab
                if key in hashes or key in donor_hashes:continue
                row={'sample_id':key,'text':modified,'length':len(modified),'article_id':article['id'],
                     'url':article['url'],'title':article['title'],'start':start,'end':finish,
                     'anchor':c,'extraction':'synthetic_substitution','synthetic':True,
                     'base_sample_id':digest(text),'base_text':text,'replacements':edits}
                emit(merged,row);emit(synthetic,row);hashes.add(key);counts.update(set(modified));lengths[len(modified)]+=1;added+=1
                donor_hashes.add(digest(text))
                diversity.add(article['id'],start,finish,text)
                if added%50000==0:print(f'Created {added:,} distinct-source synthetic samples',flush=True)
    assert all(counts[c]>=args.goal for c in vocab)
    for name in ('targets.jsonl','article_splits.jsonl'):shutil.copyfile(args.base/name,out/name)
    metadata=json.loads((args.base/'manifest.json').read_text())
    metadata.update({**resume_info,'target_samples_per_entry':args.goal,'synthesis_seed':args.seed,'synthesis_base':str(args.base.resolve()),
        'diversity_version':VERSION,'unique_source_fragment_length':SHINGLE_LENGTH,'allow_source_overlap':False,
        'synthesis':'random unseen training-corpus excerpt from source index; no shared source spans or 16-character source fragments; same-class substitution; substituted-out character may be outside vocabulary',
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
