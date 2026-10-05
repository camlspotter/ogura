"""Render a small HTML comparison report from table recipes and a CNN checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .synth_table_cells import MODES, TableCellDataset
from .table_cnn import load_model, predict_image
from .table_resize import resize_image, resize_masks


def select_samples(dataset, count: int, seed: int) -> list[dict]:
    """Balance structure modes and thin (<1px)/regular strokes without rendering."""
    if count <= 0:
        raise ValueError('count must be positive')
    buckets = {}
    for index, record in enumerate(dataset.records):
        recipe = json.loads((dataset.root/record['recipe']).read_text())
        thin = min(recipe['inner_line_width'], recipe['outer_line_width']) < 1
        key = (recipe['mode'], thin)
        buckets.setdefault(key, []).append(dict(index=index, mode=recipe['mode'], thin=thin))
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    modes = list(MODES)+sorted({key[0] for key in buckets}-set(MODES))
    keys = [(mode, thin) for mode in modes for thin in (True, False)]
    selected = []
    while len(selected) < min(count, len(dataset)):
        progressed = False
        for key in keys:
            if buckets.get(key) and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def gray(values):
    return Image.fromarray(np.rint(np.clip(values, 0, 1)*255).astype(np.uint8)).convert('RGB')


def shifted_maps(maps, offset=3):
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('overlay offset must be a nonnegative integer')
    if offset == 0:
        return maps.copy()
    shifted = np.zeros_like(maps)
    if offset < maps.shape[1]:
        shifted[0, offset:, :] = maps[0, :-offset, :]
    if offset < maps.shape[2]:
        shifted[1, :, offset:] = maps[1, :, :-offset]
    return shifted


def overlay(image, maps, image_alpha=0.3):
    strength = np.maximum(maps[0], maps[1])[..., None]*0.8
    color = np.stack([maps[0], np.zeros_like(maps[0]), maps[1]], axis=-1)
    maximum = np.maximum(maps[0], maps[1])[..., None]
    color = np.divide(color, maximum, out=np.zeros_like(color), where=maximum > 0)
    rgb = np.array(image, dtype=np.float32)/255
    rgb = rgb*image_alpha + (1-image_alpha)  # Fade the source onto white, not the prediction.
    return Image.fromarray(np.rint(np.clip(rgb*(1-strength)+color*strength, 0, 1)*255).astype(np.uint8))


def error_image(predicted, target):
    difference = predicted-target
    rgb = np.stack([np.maximum(difference, 0), np.zeros_like(difference), np.maximum(-difference, 0)], axis=-1)
    return Image.fromarray(np.rint(rgb*255).astype(np.uint8))


def comparison(image, target, predicted, overlay_offset=3):
    """Native resolution; top row has input and a displaced prediction overlay."""
    width, height = image.size
    canvas = Image.new('RGB', (width*2, height+24), '#dddddd')
    draw = ImageDraw.Draw(canvas)
    panels = [
        (0, 0, image, 'Input'),
        (width, 0, overlay(image, shifted_maps(predicted, overlay_offset)),
         f'Prediction: H red +{overlay_offset}px down / V blue +{overlay_offset}px right'),
    ]
    for x, y, panel, title in panels:
        canvas.paste(panel, (x, y+24))
        draw.text((x+4, y+5), title, fill='black')
    return canvas


def overlay_description(offset):
    return (f'左が入力、右が予測の重ね合わせ。元の罫線を見せるため、'
            f'横罫線（赤）は下へ{offset}px、縦罫線（青）は右へ{offset}pxずらしています。'
            '右の元画像は白背景に30%の濃さで表示しています。'
            '表示だけをずらし、評価値は元の座標で計算しています。')


def report(dataset_path: Path, checkpoint: Path, output: Path, *, count=20, seed=20261004,
           device='cpu', font_dir=None, overlay_offset=3, max_side=1024):
    if not isinstance(overlay_offset, int) or overlay_offset < 0:
        raise ValueError('overlay offset must be a nonnegative integer')
    dataset = TableCellDataset(dataset_path, font_dir=font_dir)
    selection = select_samples(dataset, count, seed)
    if not selection:
        raise ValueError('Dataset is empty')
    model = load_model(checkpoint, device)
    output.mkdir(parents=True, exist_ok=False)
    rows, sections = [], []
    metadata = dict(status='generating', dataset=str(dataset_path.resolve()),
                    checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    count=len(selection), seed=seed, device=device, max_side=max_side, layout_version=4, overlay_offset=overlay_offset, overlay_image_alpha=0.3,
                    selection='round-robin mode x thin/regular; not an unbiased full validation score')
    (output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for number, selected in enumerate(selection):
        sample = dataset[selected['index']]
        image=resize_image(sample['image'],max_side)
        predicted = predict_image(model, image, device).numpy()
        if not np.isfinite(predicted).all():
            raise ValueError('Prediction contains nonfinite values')
        target = np.stack([np.array(sample[k], dtype=np.float32)/255 for k in ('horizontal', 'vertical')])
        if image.size!=sample["image"].size: target=resize_masks(target,image.size)
        metrics = {}
        for c, channel in enumerate(('horizontal', 'vertical')):
            p, t = predicted[c], target[c]
            pb, tb = p >= 0.5, t >= 0.5
            denom = int(pb.sum())+int(tb.sum())
            metrics[channel] = dict(mae=float(np.abs(p-t).mean()),
                                    f1_at_05=float(2*np.logical_and(pb, tb).sum()/denom) if denom else 1.0)
        filename = f'{number:03d}-comparison.png'
        comparison(image, target, predicted, overlay_offset).save(output/filename)
        recipe = sample['recipe']
        row = dict(**selected, id=sample['id'], file=filename, original_size=list(sample['image'].size), inference_size=list(image.size),
                   inner_line_width=recipe['inner_line_width'], outer_line_width=recipe['outer_line_width'],
                   degradation=recipe['degradation'], metrics=metrics)
        rows.append(row)
        label = html.escape(f"{sample['id']} | {selected['mode']} | width {recipe['inner_line_width']}/{recipe['outer_line_width']} px | {recipe['degradation']['kind']}")
        detail = html.escape(json.dumps(metrics, ensure_ascii=False))
        sections.append(f'<section><h2>{label}</h2><p>{detail}</p><a href="{filename}"><img loading="lazy" src="{filename}" alt="{label}"></a></section>')
        print(f'{number+1}/{len(selection)}: {sample["id"]}', flush=True)
    metadata.update(status='complete', samples=rows)
    (output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    header = '''<!doctype html><meta charset="utf-8"><title>Table CNN comparisons</title>
<style>body{font:14px sans-serif;margin:24px;background:#eee}section{background:white;padding:16px;margin-bottom:24px}img{width:100%;height:auto}p{overflow-wrap:anywhere}h2{font-size:18px}</style>
<h1>Table CNN comparisons</h1>
<p>各画像は原寸で保存。クリックして拡大し、細線を確認してください。</p>
<p>OVERLAY_DESCRIPTION
予測の色の濃さは確率に対応します。</p>
<p>構造と線幅で選んだ例であり、全検証データの平均ではありません。
F1は閾値0.5を使用し、薄い罫線は正解の二値化で消える場合があります。</p>'''
    (output/'index.html').write_text(header.replace('OVERLAY_DESCRIPTION', overlay_description(overlay_offset))+'\n'.join(sections), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--max-side',type=int,default=1024)
    parser.add_argument('--font-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--overlay-offset', type=int, default=3, help='Display offset in pixels: horizontal down, vertical right')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        report(args.dataset, args.checkpoint, args.output, count=args.count, seed=args.seed,
               device=args.device, font_dir=args.font_dir, overlay_offset=args.overlay_offset,max_side=args.max_side)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
