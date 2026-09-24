"""Generate candidate text-line labels with the official Synth-JDoc HTML template."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import random
import re

from fontTools.ttLib import TTFont
from jinja2 import Template
from PIL import Image, ImageDraw, ImageOps
from playwright.sync_api import sync_playwright
from .vendor import synth_jdoc_generator as upstream

ROOT = Path(__file__).resolve().parent
UPSTREAM_REVISION = '06d27a594b5680e73f3b1308a5ff86261365903d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def render_html(record, font, seed, vertical, columns, font_size):
    random.seed(seed)
    style = upstream.generate_random_style_config(is_vertical=vertical, column_count=columns,
        font_family="'LocalDocumentFont'", base_font_size=font_size, show_title=True)
    elements = [{'type': 'text', 'content': html.escape(p)} for p in record['paragraphs']]
    rendered = Template(upstream.html_template).render(style=style,
        data={'title': html.escape(record.get('title', ''))}, processed_blocks=upstream.preprocess_elements(elements, vertical))
    # Use an embedded local font, and never depend on Google Fonts at render time.
    rendered = re.sub(r'<link\b[^>]*>', '', rendered)
    font_css = "<style>@font-face{font-family:LocalDocumentFont;src:url(data:font/otf;base64," + base64.b64encode(font.read_bytes()).decode() + ");font-weight:100 900}</style>"
    return rendered.replace('</head>', font_css+'</head>'), style


def draw_review(image_path, lines, target):
    with Image.open(image_path) as src:
        image = src.convert('RGB')
    mask = Image.new('1', image.size)
    draw = ImageDraw.Draw(mask)
    for i, line in enumerate(lines, 1):
        box = line['bbox']
        draw.rectangle(box, outline=1, width=2)
        draw.text((box[0], box[1]), str(i), fill=1)
    Image.composite(ImageOps.invert(image), image, mask).save(target)


def generate(args):
    records = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    if not records or any(not r.get('id') or not r.get('paragraphs') or
                          not all(isinstance(p, str) for p in r['paragraphs']) for r in records):
        raise ValueError('Each JSONL record needs id and a non-empty list of paragraphs')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Source IDs must be unique')
    with TTFont(args.font) as font:
        cmap = font.getBestCmap()
        missing = {c for r in records for c in (r.get('title','')+''.join(r['paragraphs']))
                   if not c.isspace() and ord(c) not in cmap}
    if missing:
        raise ValueError(f'Font lacks characters: {sorted(missing)[:20]}')
    args.output.mkdir(parents=True, exist_ok=False)
    for sub in ('images', 'html', 'review', 'metadata'):
        (args.output/sub).mkdir()
    manifest = dict(status='running', label_status='candidate_needs_visual_review',
        upstream_revision=UPSTREAM_REVISION, seed=args.seed, input_sha256=sha(args.input),
        code_sha256={name: sha(ROOT/name) for name in ('synth_jdoc.py', 'synth_lines.js')},
        font_sha256=sha(args.font), font=str(args.font.resolve()), pages=[],
        note='Plain text only; no image generation, tables, noise, or automatic train/val split')
    labels = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            manifest['browser_version'] = browser.version
            for i in range(args.count):
                record = records[i % len(records)]
                vertical = i % 2 == 1 if args.orientation == 'both' else args.orientation == 'vertical'
                columns = 1 + (i // 2) % 3
                font_size = args.font_sizes[(i // 6) % len(args.font_sizes)]
                markup, style = render_html(record, args.font, args.seed+i, vertical, columns, font_size)
                name = f'synth-{i+1:06d}'
                (args.output/'html'/f'{name}.html').write_text(markup)
                # A fixed viewport avoids the negative overflow of narrow vertical viewports.
                page = browser.new_page(viewport={'width': args.width, 'height': args.height}, device_scale_factor=1)
                page.route('**/*', lambda route: route.abort())
                page.set_content(markup, wait_until='load')
                page.evaluate('document.fonts.ready')
                if not page.evaluate("document.fonts.check('20px LocalDocumentFont')"):
                    raise ValueError('Local font did not load')
                lines = page.evaluate((ROOT/'synth_lines.js').read_text())
                image_path = args.output/'images'/f'{name}.png'
                page.screenshot(path=str(image_path), full_page=True)
                page.close()
                with Image.open(image_path) as image:
                    width, height = image.size
                if not lines:
                    raise ValueError(f'{name}: no lines')
                for line in lines:
                    x0,y0,x1,y1 = line['bbox']
                    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                        raise ValueError(f'{name}: text outside image; shorten paragraphs or increase viewport')
                labels[image_path.name] = dict(img_dimensions=[height,width], img_hash=sha(image_path),
                    polygons=[[[a,b],[c,b],[c,d],[a,d]] for a,b,c,d in (line['bbox'] for line in lines)])
                entry = dict(image=image_path.name, source_id=record['id'], seed=args.seed+i,
                             orientation='vertical' if vertical else 'horizontal', columns=columns,
                             font_size=font_size, lines=len(lines), width=width, height=height)
                write_json(args.output/'metadata'/f'{name}.json', dict(**entry, style=style, line_labels=lines))
                draw_review(image_path, lines, args.output/'review'/f'{name}_bbox.png')
                manifest['pages'].append(entry)
                write_json(args.output/'manifest.json', manifest)
                print(f'{i+1}/{args.count}: {name} {entry["orientation"]}, {len(lines)} lines', flush=True)
            browser.close()
        write_json(args.output/'labels.json', labels)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(args.output/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'synth_sample.jsonl')
    parser.add_argument('--font', type=Path, default=ROOT.parents[1]/'corpus/fonts/NotoSansCJKjp-Regular.otf')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/synth-jdoc-pilot')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--orientation', choices=['both','horizontal','vertical'], default='both')
    parser.add_argument('--font-sizes', type=int, nargs='+', default=[20,16,24])
    parser.add_argument('--width', type=int, default=1200)
    parser.add_argument('--height', type=int, default=1600)
    args = parser.parse_args()
    if min(args.count,args.width,args.height,*args.font_sizes) <= 0:
        parser.error('Counts, dimensions and font sizes must be positive')
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under ogura/textdet')
    generate(args)


if __name__ == '__main__':
    main()
