"""Rare-character-first multicover selection from verified corpus candidates."""
import argparse
from pathlib import Path
from array import array
from collections import Counter
import heapq
import json
import shutil
from extract_training_text import ROOT, emit

BASE=ROOT/'data/training_text_supplemented'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', type=int, default=100)
    parser.add_argument('--output', type=Path, default=ROOT/'data/training_text_100')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'results/training_text_100')
    args = parser.parse_args()
    if args.goal < 1: parser.error('goal must be positive')
    goal, OUT, REPORT = args.goal, args.output, args.report_dir
    if (OUT/'train.jsonl').exists():raise SystemExit('Output already exists')
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    targets=list(map(json.loads,(BASE/'targets.jsonl').open()))
    chars=[r['character'] for r in targets]; ids={c:i for i,c in enumerate(chars)}
    old=json.loads((BASE/'summary.json').read_text())
    inverted=[array('I') for _ in chars]
    flat=array('H'); offsets=array('I',[0])
    for n,line in enumerate((BASE/'train.txt').open()):
        cs=sorted({ids[c] for c in line.rstrip('\n')})
        flat.extend(cs);offsets.append(len(flat))
        for c in cs:inverted[c].append(n)
        if (n+1)%1000000==0:print(f'Indexed {n+1:,} candidates',flush=True)
    total=len(offsets)-1
    assert total==old['samples']
    required=[min(goal,len(v)) for v in inverted]
    counts=[0]*len(chars); selected=bytearray(total); chosen=[]
    def members(n):return flat[offsets[n]:offsets[n+1]]
    def score(n):return sum(counts[c]<required[c] for c in members(n))
    def accept(n):
        selected[n]=1;chosen.append(n)
        for c in members(n):counts[c]+=1
    order=sorted(range(len(chars)),key=lambda c:(len(inverted[c]),c))
    for step,c in enumerate(order):
        if counts[c]>=required[c]:continue
        if len(inverted[c])<=goal:
            for n in inverted[c]:
                if not selected[n]:accept(n)
        else:
            heap=[(-score(n),n) for n in inverted[c] if not selected[n]]
            heapq.heapify(heap)
            while counts[c]<required[c]:
                oldscore,n=heapq.heappop(heap)
                actual=-score(n)
                if actual!=oldscore:
                    heapq.heappush(heap,(actual,n));continue
                accept(n)
        if step%500==0:print(f'Selection {step:,}/{len(chars):,} characters; {len(chosen):,} samples',flush=True)
    assert all(a>=b for a,b in zip(counts,required))
    before=len(chosen)
    # Removing samples cannot make another previously necessary sample removable.
    for n in reversed(chosen):
        cs=members(n)
        if all(counts[c]>required[c] for c in cs):
            selected[n]=0
            for c in cs:counts[c]-=1
    final=sum(selected)
    print(f'Selected {before:,}; after redundancy removal {final:,}',flush=True)
    metadata=json.loads((BASE/'manifest.json').read_text())
    metadata.update({'selected_from':str(BASE),'selection':'ascending candidate frequency; maximize number of currently deficient characters; reverse redundant-sample removal',
                     'target_samples_per_entry':goal,
                     'coverage_requirement':f'min({goal}, candidate sample count) per character',
                     'length_sampling':'inherited candidate lengths; coverage-based selection',
                     'limitations':'Greedy multicover, not a proven minimum. Candidate windows may share source context.'})
    for name in ('targets.jsonl','article_splits.jsonl'):shutil.copyfile(BASE/name,OUT/name)
    (OUT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    lengths=Counter(); actual=Counter()
    with (OUT/'train.jsonl').open('w') as out:
        for n,line in enumerate((BASE/'train.jsonl').open()):
            if selected[n]:
                row=json.loads(line);emit(out,row)
                lengths[row['length']]+=1;actual.update(set(row['text']))
    assert all(actual[c]==counts[i] and actual[c]>=required[i] for i,c in enumerate(chars))
    statuses=Counter()
    with (OUT/'coverage.jsonl').open('w') as coverage,(REPORT/'shortfalls.jsonl').open('w') as short:
        for i,target in enumerate(targets):
            c=chars[i];count=counts[i]
            status='met' if count>=goal else 'shortfall' if count else 'no_selected_sample'
            statuses[status]+=1
            row={'character':c,'codepoints':[f'U+{ord(c):04X}'],'sample_count':count,'shortfall':max(0,goal-count),
                 'status':status,'corpus_occurrences':target['ranking_occurrences'],'candidate_sample_count':len(inverted[i]),'required_sample_count':required[i]}
            emit(coverage,row)
            if count<goal:emit(short,row)
    summary={**metadata,'articles':old['articles'],'article_splits':old['article_splits'],'samples':final,
             'candidate_samples':total,'selected_before_pruning':before,'length_counts':dict(sorted(lengths.items())),
             'coverage_status':dict(statuses),'all_character_requirements_preserved':True}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name in ('manifest.json','summary.json','coverage.jsonl'):shutil.copyfile(OUT/name,REPORT/name)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
