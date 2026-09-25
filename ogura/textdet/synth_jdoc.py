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
from urllib.parse import quote
from collections import Counter

from fontTools.ttLib import TTFont
from jinja2 import Template
from PIL import Image, ImageDraw, ImageOps
from playwright.sync_api import sync_playwright
from .vendor import synth_jdoc_generator as upstream
from .synth_tables import render_table, TABLE_TEXT
from .synth_images import add_image, image_plan, size_float_image

ROOT = Path(__file__).resolve().parent
UPSTREAM_REVISION = '06d27a594b5680e73f3b1308a5ff86261365903d'
TEXT_COLORS = ('#244b70', '#345b40', '#743e35', '#59416d')
TITLE_STYLES = ('outline', 'dashed', 'double', 'tinted', 'dark')



def random_heading_decoration(rng):
    accent, tint = rng.choice([('#244b70', '#e4edf5'), ('#345b40', '#e7f0e5'),
                               ('#743e35', '#f5e6e1'), ('#59416d', '#eee7f4'),
                               ('#555555', '#eeeeee')])
    background_mode = rng.choice(['plain', 'light', 'dark'])
    background = {'plain': 'transparent', 'light': tint, 'dark': accent}[background_mode]
    foreground = '#ffffff' if background_mode == 'dark' else rng.choice(('#222222', *TEXT_COLORS))
    line_color = '#ffffff' if background_mode == 'dark' else accent
    frame = rng.choice(['none', 'solid', 'dashed', 'double'])
    width = rng.choice([3, 5]) if frame == 'double' else rng.choice([1, 2, 3])
    border = 'none' if frame == 'none' else f'{width}px {frame} {line_color}'
    band = rng.choice(['none', 'inline-start', 'block-end'])
    band_width = rng.choice([3, 5, 7])
    radius = rng.choice([0, .2, .4, .6])
    css = f'background:{background};color:{foreground};border:{border};border-radius:{radius}em;'
    if band != 'none':
        css += f'border-{band}:{band_width}px solid {line_color};'
    return css, dict(background=background, foreground=foreground, border=border,
                     band=band, band_width=band_width if band != 'none' else 0,
                     line_color=line_color, border_radius_em=radius)


def title_box_css(mode, seed):
    """Decorate the existing h1; its text nodes remain the label source."""
    if mode == 'upstream':
        return '', None
    if mode == 'mixed':
        decoration, config = random_heading_decoration(random.Random(seed + 5309))
        return ('<style>h1{box-sizing:border-box;margin:0 0 24px 0;padding:16px 20px;'
                + decoration + 'line-height:1.5;overflow-wrap:anywhere}</style>'), dict(kind='mixed', **config)
    colors = {'outline': ('#ffffff', '#222222', '2px solid #244b70'),
              'dashed': ('#ffffff', '#222222', '2px dashed #555555'),
              'double': ('#ffffff', '#222222', '5px double #244b70'),
              'tinted': ('#e4edf5', '#172f46', 'none'),
              'dark': ('#244b70', '#ffffff', '2px solid #244b70')}
    background, foreground, border = colors[mode]
    css = ('<style>h1{box-sizing:border-box;margin:0 0 24px 0;padding:16px 20px;'
           f'border:{border};background:{background};color:{foreground};'
           'line-height:1.5;overflow-wrap:anywhere}</style>')
    return css, dict(kind=mode, background=background, foreground=foreground, border=border)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    temporary.replace(path)


def render_html(record, font, seed, vertical, columns, font_size, *, line_height=None,
                letter_spacing=0, font_url=None, tables=False, table_position="top", title_style='upstream', colored_text=False):
    random.seed(seed)
    style = upstream.generate_random_style_config(is_vertical=vertical, column_count=columns,
        font_family="'LocalDocumentFont'", base_font_size=font_size, show_title=True)
    if colored_text:
        color_rng = random.Random(seed + 8317)
        if color_rng.random() < .3:
            style['text_color'] = color_rng.choice(TEXT_COLORS)
    if line_height is not None:
        style['line_height'] = line_height
    style['letter_spacing'] = letter_spacing
    elements = [{'type': 'text', 'content': html.escape(p)} for p in record['paragraphs']]
    rendered = Template(upstream.html_template).render(style=style,
        data={'title': html.escape(record.get('title', ''))}, processed_blocks=upstream.preprocess_elements(elements, vertical))
    levels = record.get('heading_levels', [0] * len(record['paragraphs']))
    if len(levels) != len(record['paragraphs']) or any(level not in (0, 2, 3) for level in levels):
        raise ValueError('heading_levels must align with paragraphs and contain only 0, 2, 3')
    def heading_tag(match):
        level = levels[int(match[1])]
        return f'<h{level} data-id="{match[1]}">{match[2]}</h{level}>' if level else match[0]
    rendered = re.sub(r'<p data-id="(\d+)">(.*?)</p>', heading_tag, rendered, flags=re.DOTALL)
    # Use an embedded local font, and never depend on Google Fonts at render time.
    rendered = re.sub(r'<link\b[^>]*>', '', rendered)
    font_url = font_url or ('data:font/otf;base64,' + base64.b64encode(font.read_bytes()).decode())
    font_css = ("<style>@font-face{font-family:LocalDocumentFont;src:url('" + font_url +
                "');font-weight:100 900} .content-body{letter-spacing:" + str(letter_spacing) + "em}</style>")
    if any(levels):
        font_css += ('<style>.content-body h2,.content-body h3{line-height:1.5;'
                     'text-indent:0;break-inside:avoid;break-after:avoid;'
                     'margin-block:1em .5em;padding:.3em .5em;overflow-wrap:anywhere}'
                     '.content-body h2{font-size:1.3em}.content-body h3{font-size:1.1em}')
        rng = random.Random(seed + 7309)
        style['section_headings'] = {}
        for index, level in enumerate(levels):
            if level:
                decoration, config = random_heading_decoration(rng)
                font_css += f'.content-body h{level}[data-id="{index}"]' + '{' + decoration + '}'
                style['section_headings'][str(index)] = dict(level=level, **config)
        font_css += '</style>'
    title_css, title_config = title_box_css(title_style, seed)
    font_css += title_css
    if title_config:
        style['title_box'] = title_config
    if tables:
        if vertical:
            raise ValueError('Table pilot requires horizontal writing')
        table_css, table_html, table_config = render_table(seed)
        if table_position not in ('top', 'bottom'):
            raise ValueError('Table position must be top or bottom')
        table_config['position'] = table_position
        if table_position == 'top':
            rendered = rendered.replace('</h1>', '</h1>'+table_html, 1)
        else:
            rendered = rendered.replace('</body>', table_html+'</body>', 1)
            table_css += '<style>.table-block{margin:22px 0 0}</style>'
        font_css += table_css
        style['table'] = table_config
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


def table_positions(count, seed, mode):
    positions = [mode] * count if mode != 'both' else ['top' if i % 2 == 0 else 'bottom' for i in range(count)]
    random.Random(seed + 4817).shuffle(positions)
    return positions


def fixed_page_html(markup):
    return markup.replace('</head>', '<style>html,body{width:100%;height:100%;overflow:hidden}'
        'body{display:flex;flex-direction:column}h1{flex:none}'
        '.content-body{flex:1;min-block-size:0;inline-size:100%;block-size:auto;column-fill:auto}'
        '</style></head>')


def page_source(records, start, budget, section_headings=False):
    """Join article excerpts, preserving per-paragraph attribution and natural boundaries."""
    paragraphs, sources, owners = [], [], []
    levels = []
    characters = 0
    for offset in range(len(records)):
        record = records[(start + offset) % len(records)]
        parts = ([record.get('title', '')] if offset and record.get('title') else []) + record['paragraphs']
        prefix = len(parts) - len(record['paragraphs'])
        levels.extend(([2 if offset % 2 else 3] if prefix and section_headings else [0] * prefix)
                      + record.get('heading_levels', [0] * len(record['paragraphs'])))
        paragraphs.extend(parts)
        owners.extend([record['id']] * len(parts))
        sources.append(dict(id=record['id'], title=record.get('title', ''), source=record.get('source')))
        characters += sum(map(len, parts))
        if characters >= budget:
            break
    if characters < budget:
        raise ValueError('Input corpus is too short to fill a page without repeating articles; add more records')
    return dict(id=records[start % len(records)]['id'], title=records[start % len(records)].get('title', ''),
                paragraphs=paragraphs, sources=sources, paragraph_sources=owners, heading_levels=levels)


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
    settings = {name: getattr(args, name) for name in names}
    if getattr(args, 'title_style', 'upstream') != 'upstream':
        settings['title_style'] = args.title_style
    if getattr(args, 'colored_text', False):
        settings['colored_text'] = True
    if getattr(args, 'section_headings', False):
        settings['section_headings'] = True
    if args.tables:
        settings['tables'] = True
        settings['table_position'] = args.table_position
    if getattr(args, 'image_assets', None):
        settings['image_assets'] = {p.name: sha(p) for p in sorted(args.image_assets.glob('*.png'))}
        settings['image_position'] = args.image_position
        settings['image_float_fraction'] = args.image_float_fraction
    return settings


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
    assets = sorted(args.image_assets.glob('*.png')) if getattr(args, 'image_assets', None) else []
    if getattr(args, 'image_assets', None) and not assets:
        raise ValueError('No PNG assets found in --image-assets')
    images = image_plan([p.name for p in assets], args.count, args.seed, args.height,
                        args.image_position, args.image_float_fraction) if assets else None
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
            if args.tables:
                missing.update(c for c in TABLE_TEXT+'0123456789.,%()-（）：　' if not c.isspace() and ord(c) not in cmap)
        if missing:
            raise ValueError(f'{path}: font lacks characters: {sorted(missing)[:20]}')
    plan = variation_plan(args.count, args.seed, fonts, args.font_sizes, args.line_heights,
                          args.letter_spacings, args.vertical_fraction) if args.vary_layout else None
    positions = table_positions(args.count, args.seed, args.table_position)
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
        previous_settings = dict(previous.get('settings', legacy))
        if previous_settings.get('tables'):
            previous_settings.setdefault('table_position', 'top')
        if previous_settings != settings:
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
    if assets:
        (args.output/'assets').mkdir(exist_ok=args.resume)
        for name in sorted({entry['file'] for entry in images}):
            shutil.copyfile(args.image_assets/name, args.output/'assets'/name)
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
        code_sha256={name: sha(ROOT/name) for name in ('synth_jdoc.py', 'synth_lines.js', 'synth_paginate.js', 'synth_tables.py', 'synth_images.py')},
        fonts={str(p.resolve()):sha(p) for p in fonts}, pages=[],
        role='synthetic_training_candidate', variation=args.vary_layout,
        settings=resume_settings(args), font_order=[sha(p) for p in fonts],
        fill_page=args.fill_page, partial_fraction=args.partial_fraction,
        note='Text, optional fictional tables and reviewed local images; no image generation or automatic train/val split')
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
                    record = page_source(records, i, budget, args.section_headings)
                markup, style = render_html(record, config['font'], args.seed+i, vertical, columns, font_size,
                    line_height=config['line_height'], letter_spacing=config['letter_spacing'],
                    font_url=font_urls[config['font']], tables=args.tables, table_position=positions[i],
                    title_style=args.title_style, colored_text=args.colored_text)
                if images:
                    asset = dict(images[i], src='../assets/'+quote(images[i]['file']),
                                 sha256=manifest['settings']['image_assets'][images[i]['file']])
                    markup = add_image(markup, style, asset)
                    style['image_asset'] = asset
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
                if images and not page.evaluate("[...document.images].every(img => img.complete && img.naturalWidth > 0)"):
                    raise ValueError('Local image asset did not load')
                if images and asset['layout'] == 'float':
                    asset['float_size'] = size_float_image(page, asset['height'])
                if not page.evaluate("document.fonts.check('20px LocalDocumentFont')"):
                    raise ValueError('Local font did not load')
                lines, pagination = extract_page(page, fractions[i] if args.fill_page else None)
                if args.fill_page:
                    cx0,cy0,cx1,cy1 = pagination['content_bbox']
                    body_lines = [l for l in lines if l['element_id'].isdigit()]
                    if len(body_lines) != pagination['retained_lines'] or any(
                        not (cx0-.5 <= l['bbox'][0] < l['bbox'][2] <= cx1+.5 and
                             cy0-.5 <= l['bbox'][1] < l['bbox'][3] <= cy1+.5) for l in body_lines):
                        raise ValueError(f'{name}: page layout changed after trimming')
                    # Save the trimmed DOM so opening the HTML reproduces the image.
                    (args.output/'html'/f'{name}.html').write_text(page.content())
                expected = page.locator('h1, h2, h3, p, figcaption').all_text_contents()
                if re.sub(r'\s', '', ''.join(expected)) != re.sub(r'\s', '', ''.join(l['text'] for l in lines)):
                    raise ValueError(f'{name}: extracted text does not match rendered text')
                image_path = args.output/'images'/f'{name}.png'
                if images:
                    asset_bbox = page.locator('.asset-figure img').evaluate(
                        'img => {const r=img.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom]}')
                    a,b,c,d = asset_bbox
                    if not (0 <= a < c <= args.width and 0 <= b < d <= args.height):
                        raise ValueError(f'{name}: asset outside page')
                    if any(min(c, l['bbox'][2]) > max(a, l['bbox'][0]) and
                           min(d, l['bbox'][3]) > max(b, l['bbox'][1]) for l in lines):
                        raise ValueError(f'{name}: text overlaps image asset')
                    asset['bbox'] = asset_bbox
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
                if args.tables:
                    entry['table'] = style['table']
                    entry['table_lines'] = sum(l['element_id'].startswith('table-') for l in lines)
                    if not entry['table_lines']:
                        raise ValueError(f'{name}: missing table labels')
                if images:
                    entry['image_asset'] = asset
                if pagination:
                    entry['pagination'] = pagination
                    entry['visible_characters'] = sum(len(l['text']) for l in lines)
                    visible_ids = {record['paragraph_sources'][int(l['element_id'])] for l in lines
                                   if l['element_id'].isdigit()}
                    entry['source_ids'] = [r['id'] for r in record['sources'] if r['id'] in visible_ids]
                write_json(args.output/'metadata'/f'{name}.json', dict(**entry, style=style, line_labels=lines,
                           source=record.get('source'), sources=record.get('sources'),
                           paragraph_sources=record.get('paragraph_sources'), heading_levels=record.get('heading_levels'), title=record.get('title','')))
                draw_review(image_path, lines, args.output/'review'/f'{name}_bbox.png')
                manifest['pages'].append(entry)
                write_json(args.output/'manifest.json', manifest)
                print(f'{i+1}/{args.count}: {name} {entry["orientation"]}, {len(lines)} lines', flush=True)
            browser.close()
        write_json(args.output/'labels.json', labels)
        manifest['distributions'] = {key:dict(Counter(str(p[key]) for p in manifest['pages']))
                                    for key in ('orientation','font','font_size','columns','line_height','letter_spacing')}
        manifest['unique_sources'] = len({sid for p in manifest['pages'] for sid in p.get('source_ids', [p['source_id']])})
        if args.tables:
            manifest['table_distributions'] = {key: dict(Counter(str(p['table'].get(key, 'top') if key == 'position' else p['table'][key]) for p in manifest['pages']))
                                               for key in ('border','merged','rows','font_size','header_background','position')}
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(args.output/'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-assets', type=Path, help='Directory of reviewed text-free PNG assets')
    parser.add_argument('--image-position', choices=['top', 'bottom', 'both'], default='both')
    parser.add_argument('--image-float-fraction', type=float, default=.5,
                        help='Fraction of image pages using text wrapping instead of a separate image band')
    parser.add_argument('--colored-text', action='store_true',
                        help='Use dark colored body text on approximately 30 percent of pages')
    parser.add_argument('--section-headings', action='store_true',
                        help='Style joined article titles as alternating h2/h3 headings (requires --fill-page)')
    parser.add_argument('--title-style', choices=['upstream', 'mixed', *TITLE_STYLES], default='upstream',
                        help='Optional title textbox decoration; mixed randomizes background, frame, and band')
    parser.add_argument('--table-position', choices=['top','bottom','both'], default='both',
                        help='Position of tables; both balances top and bottom across pages')
    parser.add_argument('--tables', action='store_true', help='Add a fictional table to horizontal body text')
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
    if args.section_headings and not args.fill_page:
        parser.error('--section-headings requires --fill-page')
    if args.image_assets and not args.fill_page:
        parser.error('--image-assets requires --fill-page')
    if not 0 <= args.image_float_fraction <= 1:
        parser.error('--image-float-fraction must be between 0 and 1')
    if args.tables and (not args.fill_page or (args.vary_layout and args.vertical_fraction != 0) or
                        (not args.vary_layout and args.orientation != 'horizontal')):
        parser.error('--tables requires --fill-page and horizontal pages (--vertical-fraction 0 with --vary-layout)')
    if args.vary_layout and args.orientation != 'both':
        parser.error('Use --vertical-fraction with --vary-layout')
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under ogura/textdet')
    generate(args)


if __name__ == '__main__':
    main()
