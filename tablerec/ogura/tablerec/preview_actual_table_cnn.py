"""Preview table CNN predictions on PNG images without ground truth."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .preview_table_cnn import comparison, overlay_description
from .table_cnn import load_model, predict_image
from .table_resize import resize_image


def report(images: Path, checkpoint: Path, output: Path, *, device='cpu',
           count=None, overlay_offset=3, max_side=None, scale=None):
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError('scale must be finite and positive')
    if scale is not None and max_side is not None:
        raise ValueError('scale and max-side cannot be used together')
    if count is not None and count <= 0:
        raise ValueError('count must be positive')
    if max_side is not None and max_side <= 0:
        raise ValueError('max-side must be positive')
    if overlay_offset < 0:
        raise ValueError('overlay-offset must be nonnegative')
    if not images.is_dir():
        raise ValueError(f'Image directory not found: {images}')
    paths = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() == '.png')
    if count is not None:
        paths = paths[:count]
    if not paths:
        raise ValueError(f'No PNG images found: {images}')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    model = load_model(checkpoint, device)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(status='generating', images=str(images.resolve()),
                    checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    count=len(paths), device=device, max_side=max_side, scale=scale,
                    overlay_offset=overlay_offset, overlay_image_alpha=0.3,
                    ground_truth=False, samples=[])
    summary = output/'summary.json'
    summary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    sections = []
    for number, path in enumerate(paths):
        with Image.open(path) as source:
            rgba = ImageOps.exif_transpose(source).convert('RGBA')
            image = Image.new('RGBA', rgba.size, 'white')
            image.alpha_composite(rgba)
            image = image.convert('RGB')
        original_size = image.size
        if scale is not None:
            size = tuple(max(1, round(length*scale)) for length in image.size)
            image = image.resize(size, Image.Resampling.LANCZOS)
        if max_side is not None:
            image = resize_image(image,max_side)
        predicted = predict_image(model, image, device).numpy()
        if not np.isfinite(predicted).all():
            raise ValueError(f'Prediction contains nonfinite values: {path}')
        filename = f'{number:03d}-comparison.png'
        comparison(image, None, predicted, overlay_offset).save(output/filename)
        metadata['samples'].append(dict(source=path.name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_size=original_size, inference_size=image.size, file=filename))
        label = html.escape(path.name)
        detail = f'{original_size[0]} × {original_size[1]} → {image.width} × {image.height} px'
        sections.append(f'<section><h2>{label}</h2><p>{detail}</p>'
                        f'<a href="{filename}"><img loading="lazy" src="{filename}" alt="{label}"></a></section>')
        print(f'{number+1}/{len(paths)}: {path.name}', flush=True)
    description = overlay_description(overlay_offset).replace(
        '表示だけをずらし、評価値は元の座標で計算しています。', '表示だけをずらしています。')
    header = '''<!doctype html><meta charset="utf-8"><title>Actual table CNN previews</title>
<style>body{font:14px sans-serif;margin:24px;background:#eee}section{background:white;padding:16px;margin-bottom:24px}img{width:100%;height:auto}p,h2{overflow-wrap:anywhere}h2{font-size:18px}</style>
<h1>実画像の罫線予測</h1>'''
    (output/'index.html').write_text(header+f'<p>{description}</p>'
        '<p>正解データがないため、精度指標は計算していません。色の濃さは予測確率です。'
        '画像をクリックして拡大できます。サイズは元画像 → 推論画像で表示しています。</p>'
        +'\n'.join(sections), encoding='utf-8')
    metadata['status'] = 'complete'
    summary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--count', type=int, help='Process the first N filenames (default: all)')
    resize = parser.add_mutually_exclusive_group()
    resize.add_argument('--scale', type=float, help='Uniform image scale before inference, e.g. 0.5')
    resize.add_argument('--max-side', type=int, help='Optionally shrink before inference; default: original resolution')
    parser.add_argument('--overlay-offset', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        report(args.images, args.checkpoint, args.output, device=args.device,
               count=args.count, overlay_offset=args.overlay_offset, max_side=args.max_side, scale=args.scale)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
