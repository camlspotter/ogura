"""Evaluate the six fixed validation sets; reserved test data is never loaded."""
import argparse
import json
from pathlib import Path
import torch
from ogura.text_common import ROOT
from ogura.evaluate_lengths import load_validation_context, validation_sets, evaluate_sets
from ogura.training.aliases import CharacterAliases
from ogura.training.model import make_model
from ogura.training.train import sha256, selection_score


def suite_paths(base, suite):
    return [base/name/'validation.txt' for name in ('validation','english','validation_short5','validation_long80')] + [suite/'validation'/name/'validation.txt' for name in ('quotes','homoglyphs')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--font-dir',type=Path)
    p.add_argument('--run-config',type=Path)
    p.add_argument('--base',type=Path,default=ROOT/'datasets/japanese_english_quotes')
    p.add_argument('--suite',type=Path,default=ROOT/'datasets/evaluation_v2')
    p.add_argument('--selection-metric',choices=['mean-set-cer','mean-set-weighted-cer'],default='mean-set-cer')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.batch_size,args.threads)<1:raise ValueError('Batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(torch.cuda.current_device())
    state,vocabulary,config=load_validation_context(args.checkpoint,args.run_config,args.font_dir)
    datasets=validation_sets(config,vocabulary,suite_paths(args.base,args.suite),state['identity'])
    model=make_model(len(vocabulary),state['channels'],state['model_type']).to(device)
    model.load_state_dict(state['model'])
    rows=evaluate_sets(model,datasets,vocabulary,args.batch_size,device,CharacterAliases(state['identity'].get('evaluation_aliases')))
    aug=[r for r in rows if r['mode']=='augmented']
    score=selection_score(args.selection_metric,aug[0],[dict(r,kind='validation_length') for r in aug[1:]])
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),epoch=state['epoch'],step=state['step'],selection_metric=args.selection_metric,selection_score=score,sets=rows,suite_manifest_sha256=sha256(args.suite/'manifest.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(f'{args.selection_metric}={score:.4%}',flush=True)


if __name__=='__main__':main()
