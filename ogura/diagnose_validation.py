"""Render fixed validation errors and count aligned substitutions/deletions/insertions."""
from ogura.training.scoring import allow_punctuation_space

import argparse
from collections import Counter
from dataclasses import asdict
import heapq
import html
import json
from pathlib import Path
import time

from PIL import Image
import torch

from ogura.evaluate_lengths import load_validation_context, validation_sets
from ogura.text_common import ROOT
from ogura.training.metrics import decode, weighted_edit_distance
from ogura.training.model import make_model
from ogura.training.render import BatchRenderer
from ogura.training.train import sha256


def align_errors(reference, prediction):
    """Levenshtein alignment; ties prefer diagonal, deletion, then insertion."""
    n, m = len(reference), len(prediction)
    dp = [list(range(m+1))] + [[i] + [0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                          dp[i-1][j-1]+(reference[i-1] != prediction[j-1]))
    events = []
    i,j = n,m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1]+(reference[i-1] != prediction[j-1]):
            if reference[i-1] != prediction[j-1]:
                events.append(dict(kind='substitution',reference=reference[i-1],prediction=prediction[j-1],reference_index=i-1,prediction_index=j-1))
            i-=1;j-=1
        elif i and dp[i][j] == dp[i-1][j]+1:
            events.append(dict(kind='deletion',reference=reference[i-1],prediction='',reference_index=i-1,prediction_index=j))
            i-=1
        else:
            events.append(dict(kind='insertion',reference='',prediction=prediction[j-1],reference_index=i,prediction_index=j-1))
            j-=1
    return list(reversed(events))


def confusion_rows(counts, occurrences):
    return [dict(kind=kind, reference=ref, prediction=pred, count=count,
                 reference_codepoint=f'U+{ord(ref):04X}' if ref else None,
                 prediction_codepoint=f'U+{ord(pred):04X}' if pred else None,
                 reference_occurrences=occurrences[ref] if ref else None,
                 rate=count/occurrences[ref] if ref else None)
            for (kind,ref,pred),count in sorted(counts.items(),key=lambda item:(-item[1],item[0]))]


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_html(path, results):
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>OCR validation errors</title>',
             '<style>body{font-family:sans-serif;margin:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:none} .image{overflow-x:auto}article{border-top:1px solid #bbb;padding:12px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style>',
             '<h1>検証の誤認識</h1><p>条件ごとにCER降順。正解・予測は空白が見えるJSON表記。画像は高さ48pxの実入力（バッチの右パディングを除く）。</p>',
             '<p>混同集計は保存画像の上位例だけでなく全検証例が対象。位置は正規化後の正解の0始まり。挿入位置は文字間。曖昧な対応は対角・削除・挿入の順で決定。</p>']
    for result in results:
        info=result['condition']
        title=f"{info['dataset']} / {info['min_length']}–{info['max_length']} / {info['mode']} / {info.get('font','baseline')}"
        parts.append(f'<details open><summary>{esc(title)} — CER {result["cer"]:.4%}, weighted CER {result["weighted_cer"]:.4%}, 完全一致 {result["accuracy"]:.2%}</summary>')
        parts.append('<table><tr><th>種別</th><th>正解 → 予測</th><th>件数</th><th>正解字の出現数</th><th>率</th></tr>')
        for c in result['confusions'][:30]:
            pair=(json.dumps(c['reference'],ensure_ascii=False)+' '+(c['reference_codepoint'] or '')
                  +' → '+json.dumps(c['prediction'],ensure_ascii=False)+' '+(c['prediction_codepoint'] or ''))
            rate=f'{c["rate"]:.2%}' if c['rate'] is not None else '—'
            parts.append(f'<tr><td>{c["kind"]}</td><td>{esc(pair)}</td><td>{c["count"]}</td><td>{c["reference_occurrences"]}</td><td>{rate}</td></tr>')
        parts.append('</table>')
        for row in result['worst']:
            parts.append(f'<article><b>#{row["sample_index"]} CER {row["cer"]:.2%}, weighted CER {row["weighted_cer"]:.2%}</b><div class="image"><img src="{row["image"]}" alt="入力画像"></div>')
            for label,key in [('判定用正解','reference'),('判定用予測','prediction'),('変換前正解','raw_reference'),('変換前予測','raw_prediction')]:
                parts.append(f'<pre>{label}: {esc(json.dumps(row[key],ensure_ascii=False))}</pre>')
            parts.append('<pre>'+esc(json.dumps(row['edits'],ensure_ascii=False))+'</pre></article>')
        parts.append('</details>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def diagnose(model, datasets, vocabulary, batch_size, device, output, top=20, evaluation_aliases=None):
    from ogura.training.aliases import CharacterAliases
    evaluation_aliases = evaluation_aliases or CharacterAliases()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'images').mkdir()
    renderer=BatchRenderer(vocabulary)
    results=[]; all_counts=Counter(); all_occurrences=Counter()
    previous=model.training
    try:
        model.eval()
        with (output/'errors.jsonl').open('w',encoding='utf-8') as errors, torch.inference_mode():
            for condition_id,(info,dataset) in enumerate(datasets):
                started=time.perf_counter()
                counts=Counter();occurrences=Counter();worst=[]
                total_errors=0;weighted_errors=0.;exact=0;characters=0;accepted_by_aliases=0
                for start in range(0,len(dataset),batch_size):
                    samples=[dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]
                    batch=renderer(samples)
                    logits=model(batch.images.to(device))
                    predictions=decode(logits,model.output_lengths(batch.image_widths),vocabulary)
                    for k,(ref,pred) in enumerate(zip(batch.texts,predictions)):
                        raw_ref,raw_pred=ref,pred
                        ref,pred=evaluation_aliases.normalize(ref),evaluation_aliases.normalize(pred)
                        pred=allow_punctuation_space(ref,pred)
                        if raw_ref != raw_pred and ref == pred:accepted_by_aliases+=1
                        occurrences.update(ref);characters+=len(ref)
                        weighted = weighted_edit_distance(ref,pred);weighted_errors += weighted
                        if ref==pred:exact+=1;continue
                        edits=align_errors(ref,pred);total_errors+=len(edits)
                        counts.update((e['kind'],e['reference'],e['prediction']) for e in edits)
                        index=start+k
                        row=dict(condition_id=condition_id,sample_index=index,sample_id=samples[k].sample_id,
                                 reference=ref,prediction=pred,raw_reference=raw_ref,raw_prediction=raw_pred,cer=len(edits)/len(ref),weighted_cer=weighted/len(ref),edits=edits,
                                 render_text=samples[k].text,render_params=asdict(samples[k].render_params))
                        errors.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
                        key=(row['cer'],-index)
                        if len(worst)<top or key>worst[0][:2]:
                            width=int(batch.image_widths[k])
                            pixels=(batch.images[k,0,:,:width]*255).round().byte().numpy().copy()
                            entry=(*key,row,pixels)
                            if len(worst)<top:heapq.heappush(worst,entry)
                            else:heapq.heapreplace(worst,entry)
                kept=[]
                for rank,(_,_,row,pixels) in enumerate(sorted(worst,key=lambda e:e[:2],reverse=True),1):
                    row['image']=f'images/{condition_id:03d}-{rank:03d}.png'
                    Image.fromarray(pixels).save(output/row['image'])
                    kept.append(row)
                result=dict(condition_id=condition_id,condition=info,samples=len(dataset),
                            reference_characters=characters,character_errors=total_errors,accepted_by_aliases=accepted_by_aliases,
                            cer=total_errors/characters,weighted_character_errors=weighted_errors,weighted_cer=weighted_errors/characters,accuracy=exact/len(dataset),
                            seconds=time.perf_counter()-started,confusions=confusion_rows(counts,occurrences),
                            reference_occurrences=dict(occurrences),worst=kept)
                results.append(result);all_counts.update(counts);all_occurrences.update(occurrences)
                print(f"condition={condition_id+1}/{len(datasets)} {info['dataset']} {info['mode']} {info.get('font','')} CER={result['cer']:.4%} saved={len(kept)}",flush=True)
    finally:
        model.train(previous)
    # Repeated texts across rendering conditions count as separate observations.
    write_json(output/'confusions.json',dict(confusions=confusion_rows(all_counts,all_occurrences),reference_occurrences=dict(all_occurrences)))
    # Rendering paths may be Path objects; serialize these explicitly.
    results=json.loads(json.dumps(results,ensure_ascii=False,default=str))
    write_json(output/'conditions.json',results)
    write_html(output/'index.html',results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--run-config',type=Path)
    parser.add_argument('--evaluation-aliases',type=Path)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--validation-text',type=Path,action='append')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--top',type=int,default=20,help='Images retained per condition; statistics use all samples')
    parser.add_argument('--output',type=Path,required=True,help='New report directory')
    args=parser.parse_args()
    if min(args.batch_size,args.threads,args.top)<1:parser.error('batch-size, threads, and top must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type not in ('cpu','cuda'):raise ValueError('Use CPU or CUDA')
    if device.type=='cuda':torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    state,vocabulary,config=load_validation_context(args.checkpoint,args.run_config,args.font_dir)
    from ogura.training.aliases import CharacterAliases
    evaluation_aliases=(CharacterAliases.read(args.evaluation_aliases) if args.evaluation_aliases
                        else CharacterAliases(state['identity'].get('evaluation_aliases')))
    paths=args.validation_text or [ROOT/'datasets'/name/'validation.txt' for name in ('validation_short5','validation','validation_long80')]
    datasets=validation_sets(config,vocabulary,paths,state['identity'],all_fonts=True)
    model=make_model(len(vocabulary),state['channels'],state.get('model_type','small')).to(device)
    model.load_state_dict(state['model'])
    diagnose(model,datasets,vocabulary,args.batch_size,device,args.output,args.top,evaluation_aliases)
    write_json(args.output/'manifest.json',dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
               epoch=state['epoch'],step=state['step'],identity=state['identity'],top=args.top,
               evaluation_aliases=evaluation_aliases.config,
               alignment='Levenshtein; ties: diagonal, deletion, insertion',
               counts='All samples in all conditions; repeated texts are separate rendering observations'))
    print(f'Report: {args.output / "index.html"}')


if __name__=='__main__':main()
