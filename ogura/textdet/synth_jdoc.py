"""Generate candidate text-line labels with the official Synth-JDoc HTML template."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
from collections import Counter

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
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temporary.replace(path)


def render_html(record, font, seed, vertical, columns, font_size, *, line_height=None,
                letter_spacing=0, font_url=None):
    random.seed(seed)
    style = upstream.generate_random_style_config(is_vertical=vertical, column_count=columns,
        font_family="'LocalDocumentFont'", base_font_size=font_size, show_title=True)
    if line_height is not None:
        style['line_height'] = line_height
    style['letter_spacing'] = letter_spacing
    elements = [{'type': 'text', 'content': html.escape(p)} for p in record['paragraphs']]
    rendered = Template(upstream.html_template).render(style=style,
        data={'title': html.escape(record.get('title', ''))}, processed_blocks=upstream.preprocess_elements(elements, vertical))
    # Use an embedded local font, and never depend on Google Fonts at render time.
    rendered = re.sub(r'<link\b[^>]*>', '', rendered)
    font_url = font_url or ('data:font/otf;base64,' + base64.b64encode(font.read_bytes()).decode())
    font_css = ("<style>@font-face{font-family:LocalDocumentFont;src:url('" + font_url +
                "');font-weight:100 900} .content-body{letter-spacing:" + str(letter_spacing) + "em}</style>")
    return rendered.replace('</head>', font_css+'</head>'), style


def variation_plan(count, seed, fonts, sizes, heights, spacings, vertical_fraction):
    rng = random.Random(seed)
    def balanced(values):
        result = [values[i % len(values)] for i in range(count)]
        rng.shuffle(result)
        return result
    vertical_count = round(count * vertical_fraction)
    vertical = [True]*vertical_count + [False]*(count-vertical_count)
    rng.shuffle(vertical)
    return [dict(vertical=v, font=f, columns=c, font_size=s, line_height=h, letter_spacing=l)
            for v,f,c,s,h,l in zip(vertical, balanced(fonts), balanced([1,2,3]),
                                  balanced(sizes), balanced(heights), balanced(spacings))]


def fixed_page_html(markup):
    return markup.replace('</head>', '<style>html,body{width:100%;height:100%;overflow:hidden}'
        'body{display:flex;flex-direction:column}h1{flex:none}'
        '.content-body{flex:1;min-block-size:0;inline-size:100%;block-size:auto;column-fill:auto}'
        '</style></head>')


def page_source(records, start, budget):
    """Join article excerpts, preserving per-paragraph attribution and natural boundaries."""
    paragraphs, sources, owners = [], [], []
    characters = 0
    for offset in range(len(records)):
        record = records[(start + offset) % len(records)]
        parts = ([record.get('title', '')] if offset and record.get('title') else []) + record['paragraphs']
        paragraphs.extend(parts)
        owners.extend([record['id']] * len(parts))
        sources.append(dict(id=record['id'], title=record.get('title', ''), source=record.get('source')))
        characters += sum(map(len, parts))
        if characters >= budget:
            break
    if characters < budget:
        raise ValueError('Input corpus is too short to fill a page without repeating articles; add more records')
    return dict(id=records[start % len(records)]['id'], title=records[start % len(records)].get('title', ''),
                paragraphs=paragraphs, sources=sources, paragraph_sources=owners)


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


def resume_settings(args):
    names = ('count', 'seed', 'width', 'height', 'fill_page', 'partial_fraction',
             'vary_layout', 'vertical_fraction', 'orientation', 'font_sizes',
             'line_heights', 'letter_spacings')
    return {name: getattr(args, name) for name in names}


def restore_pages(output, manifest):
    """Recover labels from completed page metadata, without rerendering images."""
    labels = {}
    for i, entry in enumerate(manifest['pages']):
        name = f'synth-{i+1:06d}'
        if entry['image'] != name+'.png':
            raise ValueError('Resume manifest is not a contiguous page prefix')
        metadata = json.loads((output/'metadata'/f'{name}.json').read_text())
        if any(metadata.get(k) != v for k,v in entry.items()):
            raise ValueError(f'{name}: manifest/metadata mismatch')
        path = output/'images'/entry['image']
        digest = sha(path)
        if entry.get('image_sha256', digest) != digest:
            raise ValueError(f'{name}: image checksum mismatch')
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        lines = metadata['line_labels']
        if [width,height] != [entry['width'],entry['height']] or len(lines) != entry['lines']:
            raise ValueError(f'{name}: dimensions or label count mismatch')
        for line in lines:
            a,b,c,d = line['bbox']
            if not (0 <= a < c <= width and 0 <= b < d <= height):
                raise ValueError(f'{name}: invalid saved bbox')
        if not (output/'html'/f'{name}.html').is_file() or not (output/'review'/f'{name}_bbox.png').is_file():
            raise ValueError(f'{name}: missing saved HTML/review')
        labels[path.name] = dict(img_dimensions=[height,width], img_hash=digest,
            polygons=[[[a,b],[c,b],[c,d],[a,d]] for a,b,c,d in (l['bbox'] for l in lines)])
    return labels


def extract_page(page, fraction=None):
    """Keep overflow coordinates in Chromium; transfer only final page labels."""
    extract = (ROOT/'synth_lines.js').read_text()
    if fraction is None:
        return page.evaluate(extract), None
    paginate = (ROOT/'synth_paginate.js').read_text()
    script = ("fraction => { const extract = (" + extract + "); const paginate = (" + paginate +
              "); const pagination = paginate({lines: extract(), fraction});"
              " return {pagination, lines: extract()}; }")
    result = page.evaluate(script, fraction)
    return result['lines'], result['pagination']


def generate(args):
    records = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    if not records or any(not r.get('id') or not r.get('paragraphs') or
                          not all(isinstance(p, str) for p in r['paragraphs']) for r in records):
        raise ValueError('Each JSONL record needs id and a non-empty list of paragraphs')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Source IDs must be unique')
    fonts = [args.font] + args.extra_font
    for path in fonts:
        with TTFont(path) as font:
            cmap = font.getBestCmap()
            missing = {c for r in records for c in (r.get('title','')+''.join(r['paragraphs']))
                       if not c.isspace() and ord(c) not in cmap}
        if missing:
            raise ValueError(f'{path}: font lacks characters: {sorted(missing)[:20]}')
    plan = variation_plan(args.count, args.seed, fonts, args.font_sizes, args.line_heights,
                          args.letter_spacings, args.vertical_fraction) if args.vary_layout else None
    previous = None
    if args.resume:
        previous = json.loads((args.output/'manifest.json').read_text())
        settings = resume_settings(args)
        legacy = dict(count=5000, seed=20260924, width=1200, height=1600, fill_page=True,
                      partial_fraction=.2, vary_layout=True, vertical_fraction=.25,
                      orientation='both', font_sizes=[12,16,20,24], line_heights=[1.5,1.7,2.0],
                      letter_spacings=[0,.03,.08])
        if any(previous.get(key) != value for key,value in dict(seed=args.seed,
            variation=args.vary_layout, fill_page=args.fill_page, partial_fraction=args.partial_fraction,
            upstream_revision=UPSTREAM_REVISION).items()):
            raise ValueError('Resume manifest configuration differs')
        if previous.get('settings', legacy) != settings:
            raise ValueError('Resume settings differ; legacy runs only support the fixed synth_5000.sh settings')
        if previous['input_sha256'] != sha(args.input) or previous['fonts'] != {str(p.resolve()):sha(p) for p in fonts}:
            raise ValueError('Resume input/font hashes differ')
        if previous.get('settings') is None and [p.name for p in fonts] != [
            'NotoSansCJKjp-Regular.otf','NotoSansCJKjp-Bold.otf','NotoSerifCJKjp-Regular.otf','NotoSerifCJKjp-Bold.otf']:
            raise ValueError('Legacy resume requires the original font order')
        if len(previous['pages']) > args.count:
            raise ValueError('More saved pages than requested')
        restored = restore_pages(args.output, previous)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    for sub in ('images', 'html', 'review', 'metadata', 'fonts'):
        (args.output/sub).mkdir(exist_ok=args.resume)
    font_urls = {}
    for path in fonts:
        dest = args.output/'fonts'/(sha(path) + path.suffix)
        shutil.copyfile(path, dest)
        font_urls[path] = '../fonts/'+dest.name
        for license_file in path.parent.glob('LICENSE*'):
            if license_file.is_file():
                shutil.copyfile(license_file, args.output/'fonts'/license_file.name)
    manifest = dict(status='running', label_status='candidate_needs_visual_review',
        upstream_revision=UPSTREAM_REVISION, seed=args.seed, input_sha256=sha(args.input),
        code_sha256={name: sha(ROOT/name) for name in ('synth_jdoc.py', 'synth_lines.js', 'synth_paginate.js')},
        fonts={str(p.resolve()):sha(p) for p in fonts}, pages=[],
        role='synthetic_training_candidate', variation=args.vary_layout,
        settings=resume_settings(args), font_order=[sha(p) for p in fonts],
        fill_page=args.fill_page, partial_fraction=args.partial_fraction,
        note='Plain text only; no image generation, tables, noise, or automatic train/val split')
    labels = restored if previous else {}
    if previous:
        if previous.get('font_order', manifest['font_order']) != manifest['font_order']:
            raise ValueError('Resume font order differs')
        manifest['pages'] = previous['pages']
        if 'browser_version' in previous:
            manifest['browser_version'] = previous['browser_version']
        manifest['resume_history'] = previous.get('resume_history', []) + [dict(
            completed_pages=len(labels), code_sha256=previous['code_sha256'],
            legacy_images_without_checksums=any('image_sha256' not in p for p in previous['pages']))]
        print(f'Resuming after {len(labels)} verified pages', flush=True)
    fill_rng = random.Random(args.seed + 917)
    fractions = [fill_rng.uniform(.5, .85) if i < round(args.count * args.partial_fraction) else 1.0
                 for i in range(args.count)]
    fill_rng.shuffle(fractions)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            if previous and previous.get('browser_version', browser.version) != browser.version:
                raise ValueError('Resume Chromium version differs')
            manifest['browser_version'] = browser.version
            for i in range(len(labels), args.count):
                record = records[i % len(records)]
                config = plan[i] if plan else dict(
                    vertical=i % 2 == 1 if args.orientation == 'both' else args.orientation == 'vertical',
                    columns=1+(i//2)%3, font_size=args.font_sizes[(i//6)%len(args.font_sizes)],
                    font=fonts[i % len(fonts)], line_height=None, letter_spacing=0)
                vertical, columns, font_size = (config[k] for k in ('vertical','columns','font_size'))
                if args.fill_page:
                    budget = int(2 * args.width * args.height / font_size**2)
                    record = page_source(records, i, budget)
                markup, style = render_html(record, config['font'], args.seed+i, vertical, columns, font_size,
                    line_height=config['line_height'], letter_spacing=config['letter_spacing'],
                    font_url=font_urls[config['font']])
                if args.fill_page:
                    markup = fixed_page_html(markup)
                name = f'synth-{i+1:06d}'
                (args.output/'html'/f'{name}.html').write_text(markup)
                # A fixed viewport avoids the negative overflow of narrow vertical viewports.
                page = browser.new_page(viewport={'width': args.width, 'height': args.height}, device_scale_factor=1)
                # Only the saved HTML and shared local fonts can be loaded.
                allowed_root = args.output.resolve().as_uri()+'/'
                page.route('**/*', lambda route: route.continue_() if
                           route.request.url.startswith(allowed_root) else route.abort())
                page.goto((args.output/'html'/f'{name}.html').resolve().as_uri(), wait_until='load')
                page.evaluate('document.fonts.ready')
                if not page.evaluate("document.fonts.check('20px LocalDocumentFont')"):
                    raise ValueError('Local font did not load')
                lines, pagination = extract_page(page, fractions[i] if args.fill_page else None)
                if args.fill_page:
                    cx0,cy0,cx1,cy1 = pagination['content_bbox']
                    body_lines = [l for l in lines if l['element_id'] != 'title']
                    if len(body_lines) != pagination['retained_lines'] or any(
                        not (cx0-.5 <= l['bbox'][0] < l['bbox'][2] <= cx1+.5 and
                             cy0-.5 <= l['bbox'][1] < l['bbox'][3] <= cy1+.5) for l in body_lines):
                        raise ValueError(f'{name}: page layout changed after trimming')
                    # Save the trimmed DOM so opening the HTML reproduces the image.
                    (args.output/'html'/f'{name}.html').write_text(page.content())
                expected = page.locator('h1, p, figcaption').all_text_contents()
                if re.sub(r'\s', '', ''.join(expected)) != re.sub(r'\s', '', ''.join(l['text'] for l in lines)):
                    raise ValueError(f'{name}: extracted text does not match rendered text')
                image_path = args.output/'images'/f'{name}.png'
                page.screenshot(path=str(image_path), full_page=not args.fill_page)
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
                entry = dict(image=image_path.name, image_sha256=sha(image_path), source_id=record['id'], seed=args.seed+i,
                             orientation='vertical' if vertical else 'horizontal', columns=columns,
                             font_size=font_size, font=config['font'].name,
                             font_sha256=sha(config['font']), line_height=style['line_height'],
                             letter_spacing=style['letter_spacing'], paragraphs=len(record['paragraphs']),
                             characters=sum(len(p) for p in record['paragraphs']),
                             lines=len(lines), width=width, height=height)
                if pagination:
                    entry['pagination'] = pagination
                    entry['visible_characters'] = sum(len(l['text']) for l in lines)
                    visible_ids = {record['paragraph_sources'][int(l['element_id'])] for l in lines
                                   if l['element_id'] != 'title'}
                    entry['source_ids'] = [r['id'] for r in record['sources'] if r['id'] in visible_ids]
                write_json(args.output/'metadata'/f'{name}.json', dict(**entry, style=style, line_labels=lines,
                           source=record.get('source'), sources=record.get('sources'),
                           paragraph_sources=record.get('paragraph_sources'), title=record.get('title','')))
                draw_review(image_path, lines, args.output/'review'/f'{name}_bbox.png')
                manifest['pages'].append(entry)
                write_json(args.output/'manifest.json', manifest)
                print(f'{i+1}/{args.count}: {name} {entry["orientation"]}, {len(lines)} lines', flush=True)
            browser.close()
        write_json(args.output/'labels.json', labels)
        manifest['distributions'] = {key:dict(Counter(str(p[key]) for p in manifest['pages']))
                                    for key in ('orientation','font','font_size','columns','line_height','letter_spacing')}
        manifest['unique_sources'] = len({sid for p in manifest['pages'] for sid in p.get('source_ids', [p['source_id']])})
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(args.output/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Verify and retain completed pages')
    parser.add_argument('--input', type=Path, default=ROOT/'synth_sample.jsonl')
    parser.add_argument('--font', type=Path, default=ROOT.parents[1]/'corpus/fonts/NotoSansCJKjp-Regular.otf')
    parser.add_argument('--extra-font', type=Path, action='append', default=[])
    parser.add_argument('--fill-page', action='store_true', help='Join articles and retain only the first page')
    parser.add_argument('--partial-fraction', type=float, default=.2, help='Fraction of filled pages ending early')
    parser.add_argument('--vary-layout', action='store_true')
    parser.add_argument('--vertical-fraction', type=float, default=.25)
    parser.add_argument('--line-heights', type=float, nargs='+', default=[1.5,1.7,2.0])
    parser.add_argument('--letter-spacings', type=float, nargs='+', default=[0,.03,.08])
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
    if not 0 <= args.partial_fraction <= 1:
        parser.error('partial-fraction must be between 0 and 1')
    if not 0 <= args.vertical_fraction <= 1 or min(args.line_heights) <= 0 or min(args.letter_spacings) < 0:
        parser.error('Invalid vertical fraction, line height or letter spacing')
    if args.vary_layout and args.orientation != 'both':
        parser.error('Use --vertical-fraction with --vary-layout')
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under ogura/textdet')
    generate(args)


if __name__ == '__main__':
    main()
