"""Recognize cropped horizontal line images with an exported best.pt."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from ogura.training.metrics import decode
from ogura.training.model import make_model
from ogura.training.render import Vocabulary


def load_image(path):
    """Preserve aspect ratio, resize to height 48, and pad right to a multiple of 8."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        gray = Image.alpha_composite(background, rgba).convert('L')
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
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'characters' not in state:
        raise ValueError('Use an exported best.pt checkpoint')
    vocabulary = Vocabulary(state['characters'])
    model = make_model(len(vocabulary), state['channels'], state.get('model_type', 'small')).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    rows = []
    with torch.inference_mode():
        for path in args.images:
            pixels, widths = load_image(path)
            prediction = decode(model(pixels.to(device)), model.output_lengths(widths), vocabulary)[0]
            row = dict(image=str(path), prediction=prediction)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
