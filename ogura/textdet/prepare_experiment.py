"""Export reviewed pages with document/near-duplicate grouped train/val/test splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from .prepare import ROOT, write_json


def grouped_splits(pages, related_pairs, seed=20260921):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for page in pages:
        find(page['source_pdf'])
    for a, b in related_pairs:
        union(a, b)
    groups = defaultdict(list)
    for page in pages:
        groups[find(page['source_pdf'])].append(page)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(pages) * .1))
    result = {'train': [], 'val': [], 'test': []}
    for key in keys:
        split = 'val' if len(result['val']) < target else 'test' if len(result['test']) < target else 'train'
        result[split].extend(dict(p, split_group=key) for p in groups[key])
    if not all(result.values()):
        raise ValueError('Need enough independent document groups for three splits')
    return result


def export(source, output, seed=20260921):
    exclusions = json.loads((source / 'excluded_pages.json').read_text())
    excluded = {(p['source_pdf'], p['page']) for p in exclusions['pages']}
    manifest = json.loads((source / 'manifest.json').read_text())
    by_image = {p['image']: d['name'] for d in manifest['documents'] for p in d['pages']}
    # A separate experimental holdout: do not alter the user's review decisions.
    uncertain = {p['image'] for p in json.loads((source / 'blank_bbox_candidates.json').read_text())['candidates']}
    pages, held = [], []
    for doc in manifest['documents']:
        for p in doc['pages']:
            if (doc['name'], p['page']) in excluded or (source/'trash'/p['image']).exists():
                continue
            if not (source/p['image']).is_file():
                raise ValueError(f'Missing active image: {p["image"]}')
            item = dict(p, source_pdf=doc['name'])
            if p['image'] in uncertain:
                held.append(dict(item, reason='unresolved_invisible_text_or_bbox_position'))
            else:
                pages.append(item)
    similarities = json.loads((source / 'similarity_candidates.json').read_text())
    related = [(by_image[p['image_a']], by_image[p['image_b']]) for p in similarities['pairs']]
    splits = grouped_splits(pages, related, seed)
    output.mkdir(parents=True, exist_ok=False)
    report = {'seed': seed, 'source': str(source.resolve()), 'split_policy': 'PDF documents and connected similarity candidate groups stay together',
              'experimental_quality_holdout': held, 'splits': {}, 'input_sha256': {n: hashlib.sha256((source/n).read_bytes()).hexdigest()
                 for n in ['manifest.json','excluded_pages.json','similarity_candidates.json','blank_bbox_candidates.json']}}
    for split, rows in splits.items():
        directory = output/split
        (directory/'images').mkdir(parents=True)
        labels, audit = {}, []
        orientations = Counter()
        for row in rows:
            a = json.loads((source/row['labels']).read_text())
            width, height = a['image_width'], a['image_height']
            polygons = []
            for line in a['lines']:
                x0,y0,x1,y1 = line['bbox_pixels']
                x0,y0,x1,y1 = max(0,x0),max(0,y0),min(width,x1),min(height,y1)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError(f'Invalid bbox: {row["image"]}, {line["id"]}')
                polygons.append([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
                orientations[line['orientation']] += 1
            image = source/row['image']
            shutil.copyfile(image,directory/'images'/image.name)
            labels[image.name] = {'img_dimensions': [height,width], 'img_hash': hashlib.sha256(image.read_bytes()).hexdigest(), 'polygons': polygons}
            audit.append(dict(row, line_count=len(polygons), labels_sha256=hashlib.sha256((source/row['labels']).read_bytes()).hexdigest()))
        write_json(directory/'labels.json',labels)
        report['splits'][split] = {'pages': len(rows), 'documents': len({r['source_pdf'] for r in rows}), 'groups':len({r['split_group'] for r in rows}), 'boxes':sum(len(a['polygons']) for a in labels.values()), 'orientations':dict(orientations),'items':audit}
    write_json(output/'manifest.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/jdocqa-latest')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/experiment-v1')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):parser.error('Output must be under textdet')
    report=export(args.source,args.output)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='items'} for k,v in report['splits'].items()},ensure_ascii=False,indent=2))
    print('Quality holdout:',len(report['experimental_quality_holdout']))
if __name__=='__main__':main()
