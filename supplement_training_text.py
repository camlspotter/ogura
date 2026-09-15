"""Supplement deficits using only existing training-article substrings."""
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import shutil
import unicodedata as ud
import pyarrow.parquet as pq
from extract_training_text import ROOT, TOKEN, LINE, strip_han_ivs, digest, emit, split_article

BASE = ROOT / 'data/training_text_refined'
OUT = ROOT / 'data/training_text_supplemented'
REPORT = ROOT / 'results/training_text_supplemented'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'train.jsonl').exists():
        raise SystemExit('Output already exists')
    metadata = json.loads((BASE / 'manifest.json').read_text())
    old_summary = json.loads((BASE / 'summary.json').read_text())
    targets = list(map(json.loads, (BASE / 'targets.jsonl').open()))
    vocab = {r['character'] for r in targets}
    goal = metadata['target_samples_per_entry']
    counts, lengths, hashes = Counter(), Counter(), set()
    anchor_articles = Counter({r['character']: r['anchor_article_count'] for r in map(json.loads, (BASE / 'coverage.jsonl').open())})
    # Existing anchor article counts are retained; supplement-only counts are separate.
    new_anchor_articles = Counter()
    with (OUT / 'train.jsonl').open('w') as merged:
        for line in (BASE / 'train.jsonl').open():
            row = json.loads(line)
            counts.update(set(row['text']))
            lengths[row['length']] += 1
            hashes.add(row['sample_id'])
            merged.write(line)
    assert len(hashes) == old_summary['samples']
    for name in ('targets.jsonl', 'article_splits.jsonl'):
        shutil.copyfile(BASE / name, OUT / name)
    metadata.update({'supplemented_from': str(BASE), 'context': 'supplement permits repeated contexts and overlapping source intervals; exact text duplicates excluded',
        'max_samples_per_anchor_per_article': None, 'allow_source_overlap': True,
        'length_sampling': 'existing samples retained; supplement enumerates all eligible windows from length 25 down to 1 until anchor reaches goal',
        'limitations': 'Different windows can reuse one source occurrence; 500 distinct strings do not imply 500 independent contexts. Only training articles are eligible.'})
    (OUT / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    source = json.loads((ROOT / 'results/source.json').read_text())
    scanned = added = changed_ivs = 0
    active = {c for c in vocab if counts[c] < goal}
    with (OUT / 'train.jsonl').open('a') as merged, (OUT / 'additional.jsonl').open('w') as extra:
        for file in source['files']:
            path = ROOT / 'data/20231101.ja' / Path(file['path']).name
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['id','url','title','text']):
                for article in batch.to_pylist():
                    scanned += 1
                    if scanned % 1000 == 0:
                        active = {c for c in active if counts[c] < goal}
                    if split_article(article['id'], metadata['seed']) != 'train' or active.isdisjoint(article['text']):
                        continue
                    contributed = set()
                    for line in LINE.finditer(article['text']):
                        if active.isdisjoint(line.group()):
                            continue
                        matches = list(TOKEN.finditer(line.group()))
                        ts = [strip_han_ivs(m.group()) for m in matches]
                        barriers = [-1] + [i for i,c in enumerate(ts) if c not in vocab] + [len(ts)]
                        for i,char in enumerate(ts):
                            if char not in active or counts[char] >= goal:
                                continue
                            b = bisect_left(barriers, i)
                            lo, hi = barriers[b-1]+1, barriers[b]
                            for length in range(min(25,hi-lo), 0, -1):
                                for start in range(max(lo,i-length+1), min(i,hi-length)+1):
                                    end = start + length
                                    if ts[start].isspace() or ts[end-1].isspace() or ud.category(ts[start][0]).startswith('M'):
                                        continue
                                    if end < len(ts) and ud.category(ts[end][0]).startswith('M'):
                                        continue
                                    sample = ''.join(ts[start:end])
                                    key = digest(sample)
                                    if key in hashes:
                                        continue
                                    begin, finish = line.start()+matches[start].start(), line.start()+matches[end-1].end()
                                    original = article['text'][begin:finish]
                                    assert strip_han_ivs(original) == sample
                                    row = {'sample_id':key, 'text':sample, 'length':length, 'article_id':article['id'],
                                           'url':article['url'], 'title':article['title'], 'start':begin, 'end':finish,
                                           'anchor':char, 'extraction':'corpus_supplement'}
                                    emit(merged,row)
                                    emit(extra,row)
                                    hashes.add(key)
                                    counts.update(set(sample))
                                    lengths[length] += 1
                                    contributed.add(char)
                                    added += 1
                                    changed_ivs += original != sample
                                    if counts[char] >= goal:
                                        break
                                if counts[char] >= goal:
                                    break
                    new_anchor_articles.update(contributed)
            print(f'{scanned:,} articles; {added:,} added; {sum(counts[c]>=goal for c in vocab):,} targets met', flush=True)
    assert scanned == old_summary['articles']
    statuses = Counter()
    with (OUT / 'coverage.jsonl').open('w') as coverage, (REPORT / 'shortfalls.jsonl').open('w') as shortfalls:
        for target in targets:
            c=target['character']; n=counts[c]
            status='met' if n>=goal else 'shortfall' if n else 'no_selected_sample'
            statuses[status]+=1
            row={'character':c, 'codepoints':[f'U+{ord(c):04X}'], 'sample_count':n, 'shortfall':max(0,goal-n),
                 'status':status, 'corpus_occurrences':target['ranking_occurrences'],
                 'base_anchor_article_count':anchor_articles[c], 'supplement_anchor_article_count':new_anchor_articles[c]}
            emit(coverage,row)
            if n<goal:emit(shortfalls,row)
    summary={**metadata, 'articles':scanned, 'article_splits':old_summary['article_splits'],
             'samples':len(hashes), 'base_samples':old_summary['samples'], 'added_samples':added,
             'additional_ivs_normalized_samples':changed_ivs,
             'length_counts':dict(sorted(lengths.items())), 'coverage_status':dict(statuses)}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name in ('coverage.jsonl','summary.json','manifest.json'):
        shutil.copyfile(OUT/name,REPORT/name)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':main()
