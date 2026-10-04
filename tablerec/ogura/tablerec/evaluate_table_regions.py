"""Compare checkpoints on text, whitespace and true boundaries of JSON recipes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .synth_table_cells import TableCellDataset
from .table_cnn import load_model, predict_image


def regions(sample, font_dir=None):
    recipe = sample['recipe']
    text = Image.new('L', sample['image'].size)
    draw = ImageDraw.Draw(text)
    if recipe.get('background_context') != 'blank':
        path = Path(font_dir)/recipe['font']['name'] if font_dir else Path(recipe['font']['path'])
        for cell in recipe['cells']:
            font = ImageFont.truetype(str(path), cell['font_size'])
            for run in cell['text_runs']:
                x,y = run['xy']
                a,b,c,d = font.getbbox(run['text'],anchor='lt')
                draw.rectangle((x+a-1,y+b-1,x+c+1,y+d+1),fill=255)
    truth = np.stack([np.array(sample[k]) for k in ('horizontal','vertical')])/255
    border = Image.fromarray((truth.max(0)>0).astype('uint8')*255).filter(ImageFilter.MaxFilter(9))
    safe = np.array(border)==0
    text_region = (np.array(text)>0)&safe
    # White background only; avoid treating symbols or shaded cell fills as blank.
    white = (np.array(sample['image']).min(2)>=250)&safe&(np.array(text)==0)
    return truth, {'text':text_region,'white':white}


def measure(probabilities, truth, masks):
    values = {}
    for channel,name in enumerate(('horizontal','vertical')):
        p = probabilities[channel]
        for region,mask in masks.items():
            n = int(mask.sum())
            values[f'{name}/{region}'] = dict(pixels=n,
                mean_probability=float(p[mask].mean()) if n else None,
                fraction_at_05=float((p[mask]>=.5).mean()) if n else None)
        mask = truth[channel]>=.5
        n = int(mask.sum())
        values[f'{name}/boundary'] = dict(pixels=n,
            recall_at_05=float((p[mask]>=.5).mean()) if n else None)
    return values


def evaluate(dataset, checkpoints, device, count):
    report = dict(dataset=str(dataset.root.resolve()),count=min(count,len(dataset)),models=[])
    for checkpoint in checkpoints:
        model = load_model(checkpoint,device)
        samples=[]
        totals={}
        for index in range(report['count']):
            sample=dataset[index]
            truth,masks=regions(sample,dataset.font_dir)
            metrics=measure(predict_image(model,sample['image'],device).numpy(),truth,masks)
            samples.append(dict(id=sample['id'],template=sample['recipe'].get('template'),metrics=metrics))
            for key,values in metrics.items():
                accumulator=totals.setdefault(key,dict(pixels=0))
                accumulator['pixels']+=values['pixels']
                for metric,value in values.items():
                    if metric!='pixels' and value is not None:
                        accumulator[metric]=accumulator.get(metric,0)+value*values['pixels']
        for values in totals.values():
            for key in list(values):
                if key!='pixels': values[key]/=values['pixels']
        report['models'].append(dict(checkpoint=str(checkpoint.resolve()),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),metrics=totals,samples=samples))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--count',type=int,default=100)
    args=parser.parse_args()
    if args.count<=0: parser.error('--count must be positive')
    if args.output.exists(): parser.error('Output already exists')
    torch.set_num_threads(4)
    result=evaluate(TableCellDataset(args.data,font_dir=args.font_dir),args.checkpoint,args.device,args.count)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file: json.dump(result,file,indent=2)
    for model in result['models']: print(json.dumps(dict(checkpoint=model['checkpoint'],metrics=model['metrics'])))


if __name__=='__main__':
    main()
