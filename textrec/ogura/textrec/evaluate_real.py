"""Compare saved OCR predictions against visually reviewed real-image references."""
from ogura.textrec.training.scoring import allow_punctuation_space

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ogura.textrec.diagnose_validation import align_errors
from ogura.textrec.training.aliases import CharacterAliases

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def score(references, predictions, model_aliases, evaluation_aliases, preprocessing='contrast'):
    refs = {}
    for row in references:
        key = Path(row['image']).name
        if key in refs: raise ValueError(f'Duplicate reference: {key}')
        if row.get('reference_verified') is not True: raise ValueError(f'Unverified reference: {key}')
        if not row['reference'] or row['reference'] != row['reference'].strip(' \u3000'):
            raise ValueError(f'Invalid reference or edge whitespace: {key}')
        refs[key] = row
    if not refs: raise ValueError('No references')
    preds = {}
    for row in predictions:
        if row.get('preprocessing', 'none') != preprocessing: continue
        key = Path(row['image']).name
        if key in preds: raise ValueError(f'Duplicate prediction for mode: {key}')
        if not isinstance(row['prediction'], str) or any(c in row['prediction'] for c in '\r\n'):
            raise ValueError('Use the original JSONL, not wrapped terminal output')
        preds[key] = row
    if set(refs) != set(preds):
        raise ValueError(f'Image set mismatch: missing={sorted(set(refs)-set(preds))}, extra={sorted(set(preds)-set(refs))}')
    normalize = lambda t: evaluation_aliases.normalize(model_aliases.normalize(t))
    rows = []; confusions = Counter()
    for key, row in refs.items():
        ref, pred = normalize(row['reference']), normalize(preds[key]['prediction'])
        pred = allow_punctuation_space(ref, pred)
        if not ref: raise ValueError('Normalization produced empty reference')
        edits = align_errors(ref, pred)
        confusions.update((e['kind'], e['reference'], e['prediction']) for e in edits)
        rows.append(dict(image=key, raw_reference=row['reference'], raw_prediction=preds[key]['prediction'],
                         reference=ref, prediction=pred, edits=edits, errors=len(edits),
                         reference_characters=len(ref), cer=len(edits)/len(ref), notes=row.get('notes', '')))
    errors = sum(r['errors'] for r in rows); chars = sum(r['reference_characters'] for r in rows)
    return dict(samples=len(rows), exact_matches=sum(r['errors']==0 for r in rows),
                accuracy=sum(r['errors']==0 for r in rows)/len(rows), errors=errors,
                reference_characters=chars, cer=errors/chars,
                space_errors=sum(n for (_,a,b),n in confusions.items() if a==' ' or b==' '),
                confusions=[dict(kind=k,reference=a,prediction=b,count=n)
                            for (k,a,b),n in confusions.most_common()],
                rows=sorted(rows, key=lambda r:(-r['cer'],r['image'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',type=Path,default=ROOT/'datasets/real_samples/references.jsonl')
    p.add_argument('--character-aliases',type=Path,default=ROOT/'config/character_aliases.json')
    p.add_argument('--evaluation-aliases',type=Path,default=ROOT/'config/evaluation_aliases.json')
    p.add_argument('--preprocessing',choices=('none','otsu','contrast'),default='contrast')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('predictions',type=Path,nargs='+')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    refs=read_rows(args.references)
    for row in refs:
        image=args.references.parent/row['image']
        if hashlib.sha256(image.read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError(f'Reference image changed: {image}')
    aliases=CharacterAliases.read(args.character_aliases); evaluation=CharacterAliases.read(args.evaluation_aliases)
    results=[]
    for path in args.predictions:
        result=score(refs,read_rows(path),aliases,evaluation,args.preprocessing)
        result.update(predictions=str(path),predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(result)
        print(f'{path}: accuracy={result["accuracy"]:.2%} CER={result["cer"]:.4%} errors={result["errors"]}/{result["reference_characters"]} space_errors={result["space_errors"]}')
        for row in result['rows']:
            if row['errors']:
                print(f'  {row["image"]} CER={row["cer"]:.2%}\n    正解: {row["reference"]!r}\n    予測: {row["prediction"]!r}')
    comparisons=[]
    if len(results)>1:
        first={r['image']:r for r in results[0]['rows']}
        for result in results[1:]:
            comparisons.append(dict(baseline=results[0]['predictions'],candidate=result['predictions'],
                error_delta=result['errors']-results[0]['errors'],
                rows=[dict(image=r['image'],old_errors=first[r['image']]['errors'],new_errors=r['errors'],
                           delta=r['errors']-first[r['image']]['errors']) for r in result['rows']]))
    report=dict(references_sha256=hashlib.sha256(args.references.read_bytes()).hexdigest(),
                character_aliases=aliases.config,evaluation_aliases=evaluation.config,
                preprocessing=args.preprocessing,results=results,comparisons=comparisons)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
