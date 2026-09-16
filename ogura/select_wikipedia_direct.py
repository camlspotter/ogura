"""Index source articles and select rare-character-first directly from Wikipedia."""
import argparse
from array import array
from bisect import bisect_left
from collections import Counter
import heapq
import json
import pickle
import sqlite3
import zlib
from pathlib import Path
import unicodedata as ud
import pyarrow.parquet as pq
from ogura.text_common import ROOT, LINE, TOKEN, strip_han_ivs, digest, emit, split_article
from ogura.diversity import DiversityFilter, VERSION, SHINGLE_LENGTH
from ogura.prose_filter import is_prose

OUT=ROOT/'datasets/direct_100_len20_25_diverse'
REPORT=ROOT/'datasets/direct_100_len20_25_diverse/reports'
INDEX=ROOT/'cache/wikipedia_source_index'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal',type=int,default=100)
    parser.add_argument('--min-length',type=int,default=20)
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/direct_100_len20_25_diverse')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'datasets/direct_100_len20_25_diverse/reports')
    args=parser.parse_args()
    if args.goal<1:parser.error('goal must be positive')
    if not 1<=args.min_length<=25:parser.error('min-length must be 1..25')
    OUT,REPORT=args.output,args.report_dir
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);INDEX.mkdir(parents=True,exist_ok=True)
    if (OUT/'train.jsonl').exists():raise SystemExit('Output exists')
    targets=list(map(json.loads,(ROOT/'charset/selected/targets.jsonl').open()))
    chars=[r['character'] for r in targets];vocab=set(chars);ids={c:i for i,c in enumerate(chars)}
    source=json.loads((ROOT/'corpus/wikipedia/source.json').read_text());seed=20260915;goal=args.goal
    db=sqlite3.connect(INDEX/'articles.sqlite3')
    if not (INDEX/'complete.pkl').exists():
        db.execute('CREATE TABLE IF NOT EXISTS articles (n INTEGER PRIMARY KEY, payload BLOB NOT NULL)')
        db.execute('DELETE FROM articles');db.commit()
        inv=[array('I') for _ in chars]; splits=Counter();occ=Counter();scanned=0
        with (INDEX/'article_splits.jsonl').open('w') as sf:
            for file in source['files']:
                path=ROOT/'corpus/wikipedia/20231101.ja'/Path(file['path']).name
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['id','url','title','text']):
                    for a in batch.to_pylist():
                        n=scanned;scanned+=1;split=split_article(a['id'],seed);splits[split]+=1
                        emit(sf,{'article_id':a['id'],'split':split})
                        if split!='train':continue
                        freq=Counter(a['text'])
                        for c in freq.keys() & vocab:
                            inv[ids[c]].append(n);occ[c]+=freq[c]
                        db.execute('INSERT INTO articles VALUES (?,?)',(n,zlib.compress(json.dumps(a,ensure_ascii=False).encode(),1)))
                    db.commit()
                print(f'Indexed {scanned:,} articles',flush=True)
        with (INDEX/'complete.pkl').open('wb') as f:pickle.dump((inv,splits,occ,scanned,chars,source['revision'],seed),f)
    else:
        with (INDEX/'complete.pkl').open('rb') as f:inv,splits,occ,scanned,savedchars,revision,savedseed=pickle.load(f)
        assert savedchars==chars and revision==source['revision'] and savedseed==seed
    counts=Counter();hashes=set();chosen=[];lengths=Counter();anchor_articles=Counter()
    diversity=DiversityFilter()
    # All training articles are indexed. The rarest training-corpus characters go first.
    order=sorted(chars,key=lambda c:(occ[c],c))
    with (OUT/'train.jsonl').open('w') as training:
        for step,c in enumerate(order):
            if counts[c]>=goal:continue
            contributed=set()
            for n in inv[ids[c]]:
                if counts[c]>=goal:break
                a=json.loads(zlib.decompress(db.execute('SELECT payload FROM articles WHERE n=?',(n,)).fetchone()[0]))
                candidates={}
                for line in LINE.finditer(a['text']):
                    if c not in line.group() or not is_prose(line.group(),a['title']):continue
                    ms=list(TOKEN.finditer(line.group()));ts=[strip_han_ivs(m.group()) for m in ms]
                    barriers=[-1]+[i for i,t in enumerate(ts) if t not in vocab]+[len(ts)]
                    for i,t in enumerate(ts):
                        if t!=c:continue
                        b=bisect_left(barriers,i);lo,hi=barriers[b-1]+1,barriers[b]
                        for length in range(min(25,hi-lo),args.min_length-1,-1):
                            for start in range(max(lo,i-length+1),min(i,hi-length)+1):
                                end=start+length
                                if ts[start].isspace() or ts[end-1].isspace() or ud.category(ts[start][0]).startswith('M'):continue
                                if end<len(ts) and ud.category(ts[end][0]).startswith('M'):continue
                                text=''.join(ts[start:end]);key=digest(text)
                                if key in hashes or key in candidates or not is_prose(text,a['title']):continue
                                candidates[key]=(text,line.start()+ms[start].start(),line.start()+ms[end-1].end())
                def score(text):return sum(counts[x]<goal for x in set(text))
                heap=[(-score(v[0]),-len(v[0]),key) for key,v in candidates.items()];heapq.heapify(heap)
                while heap and counts[c]<goal:
                    previous,neglen,key=heapq.heappop(heap);text,begin,end=candidates[key]
                    if not diversity.allows(a["id"],begin,end,text):continue
                    current=-score(text)
                    if current!=previous:heapq.heappush(heap,(current,neglen,key));continue
                    assert strip_han_ivs(a['text'][begin:end])==text
                    row={'sample_id':key,'text':text,'length':len(text),'article_id':a['id'],'url':a['url'],'title':a['title'],
                         'start':begin,'end':end,'anchor':c,'extraction':'direct_rare_first'}
                    diversity.add(a["id"],begin,end,text)
                    emit(training,row);hashes.add(key);counts.update(set(text));lengths[len(text)]+=1;contributed.add(n)
            anchor_articles[c]=len(contributed)
            if step%500==0:print(f'Selected for {step:,}/{len(chars):,} characters; {len(hashes):,} samples',flush=True)
    import shutil
    shutil.copyfile(ROOT/'charset/selected/targets.jsonl',OUT/'targets.jsonl')
    shutil.copyfile(INDEX/'article_splits.jsonl',OUT/'article_splits.jsonl')
    metadata={'seed':seed,'target_samples_per_entry':goal,'target_entries':len(chars),'target_sha256':__import__('hashlib').sha256((OUT/'targets.jsonl').read_bytes()).hexdigest(),
        'source_repository':source['repository'],'source_revision':source['revision'],'source_config':source['config'],
        'strip_han_ivs':True,'allow_source_overlap':False,'diversity_version':VERSION,'unique_source_fragment_length':SHINGLE_LENGTH,'max_samples_per_anchor_per_article':None,'max_tokens':25,'min_tokens':args.min_length,
        'prose_filter':'Japanese grammar evidence and >=3 hiragana; reject kana charts and character-table titles',
        'selection':'ascending training-corpus character frequency; source articles in corpus order; within each article greedily maximize deficient character coverage, then length',
        'normalization':'remove supplementary Han IVS only; offsets refer to original article',
        'limitations':'Greedy, not a minimum. Overlapping source spans and repeated 16-character source fragments are excluded. No synthesized text. Validation/test articles excluded.'}
    statuses=Counter()
    with (OUT/'coverage.jsonl').open('w') as f,(REPORT/'shortfalls.jsonl').open('w') as short:
        for r in targets:
            c=r['character'];num=counts[c];status='met' if num>=goal else 'shortfall' if num else 'no_selected_sample';statuses[status]+=1
            row={'character':c,'sample_count':num,'shortfall':max(0,goal-num),'status':status,'corpus_occurrences':r['ranking_occurrences'],
                'training_corpus_occurrences':occ[c],'training_article_count':len(inv[ids[c]]),'anchor_article_count':anchor_articles[c]}
            emit(f,row)
            if num<goal:emit(short,row)
    summary={**metadata,'samples':len(hashes),'articles':scanned,'article_splits':dict(splits),'length_counts':dict(sorted(lengths.items())),'coverage_status':dict(statuses)}
    for name,obj in [('manifest.json',metadata),('summary.json',summary)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    for name in ('manifest.json','summary.json','coverage.jsonl'):shutil.copyfile(OUT/name,REPORT/name)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
