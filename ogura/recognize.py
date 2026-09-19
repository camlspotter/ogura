"""Recognize cropped horizontal line images with an exported best.pt."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from ogura.training.metrics import decode
from ogura.training.model import make_model
from ogura.training.render import Vocabulary


def binarize(gray):
    """Otsu threshold on the original resolution; dark foreground stays black."""
    histogram = np.bincount(np.asarray(gray).ravel(), minlength=256).astype(np.float64)
    weight = histogram.cumsum()
    moment = (histogram * np.arange(256)).cumsum()
    denominator = weight * (weight[-1] - weight)
    score = np.full(256, -1.0)
    valid = denominator > 0
    score[valid] = (moment[-1] * weight[valid] - moment[valid] * weight[-1])**2 / denominator[valid]
    if not valid.any():
        return gray.copy()
    threshold = int(score.argmax())
    return gray.point(lambda value: 0 if value <= threshold else 255)


def adjust_contrast(gray, cutoff=1.0):
    """Stretch grayscale levels; trim cutoff percent from each histogram end."""
    if not math.isfinite(cutoff) or not 0 <= cutoff < 50:
        raise ValueError('Contrast cutoff must be finite and in [0, 50)')
    return ImageOps.autocontrast(gray, cutoff=cutoff)


def load_image(path, preprocessing='none', contrast_cutoff=1.0):
    """Preserve aspect ratio, resize to height 48, and pad right to a multiple of 8."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        gray = Image.alpha_composite(background, rgba).convert('L')
    if preprocessing == 'otsu':
        gray = binarize(gray)
    elif preprocessing == 'contrast':
        gray = adjust_contrast(gray, contrast_cutoff)
    elif preprocessing != 'none':
        raise ValueError(f'Unknown preprocessing: {preprocessing}')
    width = max(1, round(gray.width * 48 / gray.height))
    gray = gray.resize((width, 48), Image.Resampling.LANCZOS)
    pixels = torch.ones(1, 1, 48, (width + 7)//8*8)
    pixels[0, 0, :, :width] = torch.from_numpy(np.array(gray, dtype=np.float32)/255)
    return pixels, torch.tensor([width], dtype=torch.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preprocessing', choices=('none', 'otsu', 'contrast'), default='none')
    parser.add_argument('--compare-preprocessing', action='store_true',
                        help='Compare none, Otsu binarization, and grayscale contrast stretching')
    parser.add_argument('--contrast-cutoff', type=float, default=1.0,
                        help='Percent clipped at each histogram end for contrast mode (default: 1)')
    parser.add_argument('--save-inputs', type=Path,
                        help='New directory for the actual resized model inputs')
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.contrast_cutoff) or not 0 <= args.contrast_cutoff < 50:
        parser.error('--contrast-cutoff must be finite and in [0, 50)')
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if args.save_inputs:
        args.save_inputs.mkdir(parents=True, exist_ok=False)
    modes = ('none', 'otsu', 'contrast') if args.compare_preprocessing else (args.preprocessing,)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'characters' not in state:
        raise ValueError('Use an exported best.pt checkpoint')
    vocabulary = Vocabulary(state.get('source_characters', state['characters']),
                            state.get('identity', {}).get('character_aliases'))
    model = make_model(len(vocabulary), state['channels'], state.get('model_type', 'small')).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(args.images, 1):
            for mode in modes:
                pixels, widths = load_image(path, mode, args.contrast_cutoff)
                if args.save_inputs:
                    Image.fromarray((pixels[0, 0].numpy()*255).round().astype(np.uint8)).save(
                        args.save_inputs/f'{index:03d}-{path.stem}-{mode}.png')
                prediction = decode(model(pixels.to(device)), model.output_lengths(widths), vocabulary)[0]
                row = dict(image=str(path), preprocessing=mode, prediction=prediction)
                if mode == 'contrast':
                    row['contrast_cutoff'] = args.contrast_cutoff
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
