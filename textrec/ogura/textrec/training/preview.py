"""Render one text under varied training conditions into a single contact sheet."""
from ogura.textrec.paths import CORPUS_ROOT

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ogura.textrec.text_common import ROOT
from .render import RenderParams, Sample, load_font, parameters_for_sample, render_sample, normalized_text


def create_preview(text, fonts, output, seed=20260915, variants=3,
                   size_min=28, size_max=40, padding_min=2, padding_max=6, vertical_jitter=3,
                   vertical_mode="full", western_fonts=()):
    if not fonts or variants < 1:
        raise ValueError('Provide fonts and at least one variant per font')
    fonts = [Path(p).resolve() for p in fonts]
    western_fonts = [Path(p).resolve() for p in western_fonts]
    for font in [*fonts, *western_fonts]:
        if not font.is_file():
            raise FileNotFoundError(f'{font}: run python -m ogura.textrec.download_training_fonts first')
    sample_id = hashlib.sha256(text.encode()).hexdigest()
    rows = [('Baseline', RenderParams(str(fonts[0])))]
    # Stratify by font so even a small sheet shows every requested family/weight.
    # Each variant uses the exact training parameter sampler and renderer.
    for font in fonts:
        for variant in range(variants):
            p = parameters_for_sample(font, seed, variant, sample_id,
                                      size_min, size_max, padding_min=padding_min,
                                      padding_max=padding_max, vertical_jitter=vertical_jitter,
                                      vertical_full_range=vertical_mode == "full")
            for western in western_fonts or [None]:
                rows.append((font.stem + (f" + {western.stem}" if western else ""),
                             replace(p, western_font_path=str(western) if western else None)))
    rendered = [render_sample(Sample(text, p, sample_id)) for _, p in rows]
    label_font = load_font(str(fonts[0]), 15)
    title_font = load_font(str(fonts[0]), 22)
    labels = []
    metadata = []
    for (name, p), im in zip(rows, rendered):
        left = p.padding if p.padding_left is None else p.padding_left
        right = p.padding if p.padding_right is None else p.padding_right
        placement = f'Y range={p.vertical_position:.0%}' if p.vertical_position is not None else f'dy={p.vertical_offset:+d}px'
        label = (f'{name}  |  size request={p.font_size}px  L/R={left}/{right}px  '
                 f'Y min={p.padding}px  {placement}  |  {im.width} x {im.height}px')
        labels.append(label)
        metadata.append(dict(name=name, parameters=asdict(p), width=im.width, height=im.height,
                             rendered_text=normalized_text(text, p)))
    margin, row_height, header = 24, 90, 102
    width = max(960, max(im.width for im in rendered)+margin*2,
                int(max(label_font.getlength(s) for s in labels))+margin*2,
                int(title_font.getlength(text))+margin*2)
    sheet = Image.new('RGB', (width, header + len(rows)*row_height + margin), '#edf0f4')
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), text, font=title_font, fill='#172334')
    draw.text((margin, 47), '48px height / original pixel size / same text in every row', font=label_font, fill='#33445c')
    draw.text((margin, 69), 'Y range: 0%=top, 100%=bottom within safe margins. White area = input image.', font=label_font, fill='#33445c')
    for index, (label, im) in enumerate(zip(labels, rendered)):
        y = header + index*row_height
        draw.text((margin, y), label, font=label_font, fill='#33445c')
        sheet.paste(im, (margin, y+27))
        draw.rectangle((margin-1, y+26, margin+im.width, y+27+im.height), outline='#aab5c5')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format='PNG')
    output.with_suffix('.json').write_text(json.dumps(dict(text=text, seed=seed, rows=metadata), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default='春の図書館で、日本語の文字をゆっくり読む。')
    parser.add_argument('--western-font', type=Path, action='append', default=[])
    parser.add_argument('--font', type=Path, action='append', help='Repeat to choose fonts; default: four Noto fonts')
    parser.add_argument('--output', type=Path, default=ROOT/'previews/augmentation.png')
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--variants', type=int, default=3, help='Variants per font, plus one baseline')
    for name, default in (('size-min',28), ('size-max',40), ('padding-min',2), ('padding-max',6), ('vertical-jitter',3)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument("--vertical-mode", choices=("full", "jitter"), default="full")
    args = parser.parse_args()
    fonts = args.font or [CORPUS_ROOT / 'fonts'/f'Noto{family}CJKjp-{weight}.otf'
                          for family in ('Sans','Serif') for weight in ('Regular','Bold')]
    print(create_preview(args.text, fonts, args.output, args.seed, args.variants,
                         args.size_min, args.size_max, args.padding_min, args.padding_max, args.vertical_jitter, args.vertical_mode, args.western_font))


if __name__ == '__main__':
    main()
