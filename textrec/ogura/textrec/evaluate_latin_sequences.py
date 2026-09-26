"""Compare checkpoints on identical images, including per-sequence deletion rates."""
from ogura.textrec.paths import CORPUS_ROOT

from ogura.textrec.training.scoring import allow_punctuation_space

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from ogura.textrec.collect_latin_sequences import PATTERNS, WORD
from ogura.textrec.diagnose_validation import align_errors
from ogura.textrec.evaluate_lengths import load_validation_context, validation_sets
from ogura.textrec.training.aliases import CharacterAliases
from ogura.textrec.training.metrics import decode, weighted_edit_distance
from ogura.textrec.training.model import make_model
from ogura.textrec.training.render import BatchRenderer
from ogura.textrec.build_english import sha
from ogura.textrec.text_common import ROOT


def sequence_counts(reference, prediction):
    """Count overlapping occurrences within whole words of rendered/scored labels.

    Alignment ties use diagnose_validation's diagonal/deletion/insertion order.
    Counts describe that minimum-edit alignment, not a unique physical location
    of missing ink. Repeated letters can admit multiple equally good alignments.
    Adjacent patterns overlap, so their denominators must never be summed as a
    count of independent samples. Insertions are covered by overall CER.
    """
    edits=align_errors(reference,prediction)
    deleted={e['reference_index'] for e in edits if e['kind']=='deletion'}
    substituted={e['reference_index'] for e in edits if e['kind']=='substitution'}
    result={p:Counter() for p in PATTERNS}
    for word in WORD.finditer(reference):
        for p in PATTERNS:
            for i in range(word.start(),word.end()-len(p)+1):
                if reference[i:i+len(p)]!=p:continue
                span=set(range(i,i+len(p)));d=len(span&deleted);s=len(span&substituted)
                result[p].update(occurrences=1,reference_characters=len(p),deleted_occurrences=int(d>0),
                                 deleted_characters=d,substituted_occurrences=int(s>0),substituted_characters=s)
    return result,edits


def rates(counts):
    return {p:dict(c,deletion_rate=c.get('deleted_occurrences',0)/c['occurrences'] if c.get('occurrences') else None,
                   deleted_character_rate=c.get('deleted_characters',0)/c['reference_characters'] if c.get('reference_characters') else None)
            for p,c in counts.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--validation-text',type=Path,default=ROOT/'datasets/validation_latin_sequences/validation.txt')
    p.add_argument('--font-dir',type=Path,default=CORPUS_ROOT / 'fonts')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if min(a.batch_size,a.threads)<1:raise ValueError('Batch size and threads must be positive')
    if len(set(a.checkpoint))!=len(a.checkpoint):raise ValueError('Duplicate checkpoints')
    torch.set_num_threads(a.threads);device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(torch.cuda.current_device())
    contexts=[load_validation_context(c,font_dir=a.font_dir) for c in a.checkpoint]
    identity=contexts[0][0]['identity']
    if any(state['identity']!=identity for state,_,_ in contexts):
        raise ValueError('Use checkpoints from the same run/settings for an identical-image comparison')
    state,vocabulary,config=contexts[0]
    datasets=validation_sets(config,vocabulary,[a.validation_text],identity)
    aliases=CharacterAliases(identity.get('evaluation_aliases'))
    models=[]
    for state,_,_ in contexts:
        model=make_model(len(vocabulary),state['channels'],state.get('model_type','small')).to(device)
        model.load_state_dict(state['model']);model.eval();models.append(model)
    a.output.mkdir(parents=True)
    renderer=BatchRenderer(vocabulary);reports=[[] for _ in models]
    started=time.perf_counter()
    # Render each batch once and feed the exact same tensor to every checkpoint.
    with (a.output/'predictions.jsonl').open('w') as log,torch.inference_mode():
        for info,dataset in datasets:
            counts=[{p:Counter() for p in PATTERNS} for _ in models]
            totals=[Counter() for _ in models]
            for start in range(0,len(dataset),a.batch_size):
                samples=[dataset[i] for i in range(start,min(start+a.batch_size,len(dataset)))]
                batch=renderer(samples);images=batch.images.to(device)
                for k,model in enumerate(models):
                    predictions=decode(model(images),model.output_lengths(batch.image_widths),vocabulary)
                    for index,(raw_ref,raw_pred) in enumerate(zip(batch.texts,predictions)):
                        ref=aliases.normalize(raw_ref);pred=aliases.normalize(raw_pred)
                        pred=allow_punctuation_space(ref,pred)
                        c,edits=sequence_counts(ref,pred)
                        for pattern in PATTERNS:counts[k][pattern].update(c[pattern])
                        totals[k].update(samples=1,exact_matches=int(ref==pred),reference_characters=len(ref),
                                         errors=len(edits),weighted_errors=weighted_edit_distance(ref,pred))
                        row=dict(checkpoint=str(a.checkpoint[k]),mode=info['mode'],sample_id=samples[index].sample_id,
                                 input_text=samples[index].text,reference=ref,prediction=pred,edits=edits,
                                 render_params=asdict(samples[index].render_params),image_width=int(batch.image_widths[index]))
                        log.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
            for k,t in enumerate(totals):
                row=dict(mode=info['mode'],**t,cer=t['errors']/t['reference_characters'],
                         weighted_cer=t['weighted_errors']/t['reference_characters'],accuracy=t['exact_matches']/t['samples'],
                         sequences=rates(counts[k]))
                reports[k].append(row)
                print(f"{a.checkpoint[k]} mode={info['mode']} accuracy={row['accuracy']:.2%} CER={row['cer']:.4%}",flush=True)
                for pattern,c in row['sequences'].items():
                    print(f"  {pattern}: deletions={c.get('deleted_occurrences',0)}/{c.get('occurrences',0)}",flush=True)
    report=dict(validation_sha256=sha(a.validation_text),manifest_sha256=sha(a.validation_text.parent/'manifest.json'),
                seconds=time.perf_counter()-started,device=a.device,torch_version=str(torch.__version__),
                scoring='Ordinary Levenshtein alignment; sequence deletion rate is affected occurrences / all occurrences; overlaps count separately.',
                checkpoints=[dict(path=str(path),sha256=sha(path),epoch=state['epoch'],step=state['step'],results=results)
                             for path,(state,_,_),results in zip(a.checkpoint,contexts,reports)])
    (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 欧文連続文字の比較','', '各値は欠落を含む出現数 / 全出現数。括弧内は割合。組の重複あり。','']
    for mode in ('baseline','augmented'):
        lines+=['## '+mode,'','|組|'+'|'.join(str(c) for c in a.checkpoint)+'|','|---|'+'---:|'*len(models)]
        selected=[next(r for r in rs if r['mode']==mode) for rs in reports]
        for pattern in PATTERNS:
            values=[]
            for row in selected:
                c=row['sequences'][pattern];n=c.get('occurrences',0);d=c.get('deleted_occurrences',0)
                values.append(f'{d}/{n} ({d/n:.2%})' if n else '対象なし')
            lines.append('|'+pattern+'|'+'|'.join(values)+'|')
        lines.append('')
    (a.output/'comparison.md').write_text('\n'.join(lines))
    print(f'Report: {a.output}/comparison.md',flush=True)


if __name__=='__main__':main()
