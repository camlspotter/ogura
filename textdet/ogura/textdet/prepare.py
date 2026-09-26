"""Filter entire PDFs, then export all page images and PDF-derived text-line boxes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps
import pymupdf as fitz

from .glyph_bounds import lookup_bounds, vertical_glyph_bounds

ROOT = Path(__file__).resolve().parents[2]
FONT_ROW = re.compile(
    r'^(.*?)\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+(\d+)\s+(\d+)\s*$'
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_fonts(output):
    """Parse pdffonts' stable rightmost columns; fail closed on unexpected rows."""
    rows = output.splitlines()
    divider = next((i for i, row in enumerate(rows) if row.startswith('---')), None)
    if divider is None:
        raise ValueError('pdffonts: missing table header')
    fonts = []
    for row in rows[divider + 1:]:
        if not row.strip():
            continue
        match = FONT_ROW.fullmatch(row)
        if not match:
            raise ValueError(f'pdffonts: unrecognized row: {row}')
        description, embedded, subset, unicode_map, obj, generation = match.groups()
        fonts.append({'description': description.strip(), 'embedded': embedded == 'yes',
                      'subset': subset == 'yes', 'to_unicode': unicode_map == 'yes',
                      'object_id': int(obj), 'generation': int(generation)})
    return fonts


def inspect_fonts(path):
    # PDF font names can contain raw legacy-encoded bytes. Keep them escaped in
    # the report without losing the ASCII yes/no mapping flags on the right.
    result = subprocess.run(['pdffonts', str(path)], capture_output=True, text=True,
                            encoding='utf-8', errors='backslashreplace', timeout=120)
    if result.returncode:
        raise ValueError(f'pdffonts failed: {result.stderr.strip()}')
    return parse_fonts(result.stdout), result.stderr.strip()


def suspicious(char):
    n = ord(char)
    return (char == '\ufffd' or n == 0 or 0xE000 <= n <= 0xF8FF or
            0xF0000 <= n <= 0xFFFFD or 0x100000 <= n <= 0x10FFFD or
            (n < 32 and not char.isspace()))


def quad_points(quad, page):
    return [[p.x, p.y] for p in [q * page.rotation_matrix
                                for q in (quad.ul, quad.ur, quad.lr, quad.ll)]]


def bounds(points):
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def merge_vertical_fragments(lines):
    """Join adjacent fragments within a PDF block, never across columns or gaps."""
    result = []
    for line in lines:
        previous = result[-1] if result else None
        merge = False
        if previous and previous['wmode'] == line['wmode'] == 1 and previous['baseline'] == line['baseline'] == (0.0, 1.0):
            a, b = previous['chars'][-1], line['chars'][0]
            size = min(a['size'], b['size'])
            merge = (size > 0 and max(a['size'], b['size']) <= size * 1.25
                     and abs(a['origin'][0] - b['origin'][0]) <= size * 0.2
                     and 0 < b['origin'][1] - a['origin'][1] <= size * 1.6)
        if not merge:
            result.append(line)
            continue
        previous['text'] += line['text']
        previous['chars'].extend(line['chars'])
        previous['fonts'] = sorted(set(previous['fonts']) | set(line['fonts']))
        previous['source_line_ids'].extend(line['source_line_ids'])
        x0, y0, x1, y1 = bounds(previous['polygon'] + line['polygon'])
        previous['bbox'] = [x0, y0, x1, y1]
        previous['polygon'] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return result


def merge_rotated_vertical_prefixes(lines, derotation_matrix):
    """Attach a narrow rotated glyph to an unambiguous column immediately below."""
    def raw_bbox(line):
        return bounds([[p.x, p.y] for p in
                       (fitz.Point(point) * derotation_matrix for point in line['polygon'])])

    proposals = {}
    for i, prefix in enumerate(lines):
        if (prefix['wmode'] != 1 or prefix['baseline'] not in ((1., 0.), (-1., 0.))
                or len(prefix['chars']) != 1
                or prefix['chars'][0].get('geometry_source') != 'glyph_outline'):
            continue
        x0, y0, x1, y1 = raw_bbox(prefix)
        if y1 - y0 < 2 * (x1 - x0):
            continue
        candidates = []
        for j, following in enumerate(lines):
            if following['wmode'] != 1 or following['baseline'] != (0., 1.):
                continue
            a, b = prefix['chars'][0]['size'], following['chars'][0]['size']
            size = min(a, b)
            if size <= 0 or max(a, b) > 1.25 * size:
                continue
            # Compare the first glyph rather than the whole column: parentheses
            # or punctuation at its end can shift the column bbox's center.
            first = dict(following, polygon=following['chars'][0]['polygon'])
            fx0, fy0, fx1, _ = raw_bbox(first)
            if (abs((x0 + x1 - fx0 - fx1) / 2) <= .25 * size
                    and 0 <= fy0 - y1 <= .6 * size):
                candidates.append(j)
        if len(candidates) == 1:
            proposals[i] = candidates[0]
    consumed = set()
    for i, j in proposals.items():
        if list(proposals.values()).count(j) != 1:
            continue
        prefix, following = lines[i], lines[j]
        prefix['text'] += following['text']
        prefix['chars'].extend(following['chars'])
        prefix['source_line_ids'].extend(following['source_line_ids'])
        prefix['fonts'] = sorted(set(prefix['fonts']) | set(following['fonts']))
        prefix['baseline'] = following['baseline']
        x0, y0, x1, y1 = bounds(prefix['polygon'] + following['polygon'])
        prefix['bbox'] = [x0, y0, x1, y1]
        prefix['polygon'] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        consumed.add(j)
    return [line for i, line in enumerate(lines) if i not in consumed]


def merge_remaining_stacked_characters(lines):
    """Last pass: infer columns only from still-isolated single characters."""
    result, run = [], []

    def flush():
        if len(run) < 3:
            result.extend(run)
        else:
            merged = dict(run[0])
            merged['text'] = ''.join(line['text'] for line in run)
            merged['chars'] = [char for line in run for char in line['chars']]
            merged['source_line_ids'] = [i for line in run for i in line['source_line_ids']]
            merged['orientation'] = 'vertical'
            merged['orientation_source'] = 'isolated_character_positions'
            # wmode and baseline retain the source PDF's horizontal glyph setting.
            x0, y0, x1, y1 = bounds([p for line in run for p in line['polygon']])
            merged['bbox'] = [x0, y0, x1, y1]
            merged['polygon'] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            result.append(merged)
        run.clear()

    for line in lines:
        if (len(line['chars']) != 1 or len(line['text']) != 1 or line['wmode'] != 0
                or tuple(line['baseline']) != (1., 0.)):
            flush()
            result.append(line)
            continue
        if run:
            previous = run[-1]
            a, b = previous['chars'][0], line['chars'][0]
            size = min(a['size'], b['size'])
            if not (size > 0 and previous['source_block'] == line['source_block']
                    and previous['fonts'] == line['fonts']
                    and max(a['size'], b['size']) <= 1.2 * size
                    and abs(run[0]['chars'][0]['origin'][0] - b['origin'][0]) <= .15 * size
                    and .5 * size <= b['origin'][1] - a['origin'][1] <= 1.4 * size):
                flush()
        run.append(line)
    flush()
    return result


def trim_edge_space_chars(chars):
    start, end = 0, len(chars)
    while start < end and not chars[start]['text'].strip():
        start += 1
    while end > start and not chars[end - 1]['text'].strip():
        end -= 1
    return chars[start:end]


def character_range_polygon(chars, direction, rotation):
    """Enclose character quads in the text's axes, including rotated text."""
    dx, dy = direction
    ux, uy = dx * rotation.a + dy * rotation.c, dx * rotation.b + dy * rotation.d
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    vx, vy = -uy, ux
    points = [point for char in chars for point in char['polygon']]
    x0, y0, x1, y1 = bounds([[x * ux + y * uy, x * vx + y * vy] for x, y in points])
    return [[x * ux + y * vx, x * uy + y * vy]
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def extract_page(page):
    # Preserve font-derived character codes; no OCR, substitution, or normalization.
    raw = page.get_text('rawdict', sort=False, flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)
    glyph_bounds = vertical_glyph_bounds(page) if any(
        l.get('wmode') == 1 for b in raw['blocks'] for l in b.get('lines', [])
    ) else {}
    lines = []
    source_line_id = 0
    for block_index, block in enumerate(raw['blocks']):
        if block['type'] != 0:
            continue
        block_lines = []
        for line in block['lines']:
            source_line_id += 1
            vertical_axis = line.get('wmode') == 1 and line['dir'] in (
                (0., 1.), (1., 0.), (0., -1.), (-1., 0.))
            chars, fonts = [], set()
            for span in line['spans']:
                fonts.add(span['font'])
                for char in span['chars']:
                    glyph_bbox = lookup_bounds(glyph_bounds, span, char) if vertical_axis else None
                    chars.append({'text': char['c'], 'font': span['font'], 'size': span['size'],
                                  'origin': char['origin'],
                                  'geometry_source': 'glyph_outline' if glyph_bbox is not None else 'pdf_font_metrics',
                                  'polygon': quad_points(glyph_bbox.quad if glyph_bbox is not None
                                                         else fitz.recover_char_quad(line['dir'], span, char), page)})
            text = ''.join(c['text'] for c in chars)
            if not text.strip():
                continue
            bbox_chars = trim_edge_space_chars(chars)
            if vertical_axis and any(c['geometry_source'] == 'glyph_outline' for c in chars):
                rect = bounds([p for c in bbox_chars for p in c['polygon']])
                x0, y0, x1, y1 = rect
                polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            elif len(bbox_chars) != len(chars):
                polygon = character_range_polygon(bbox_chars, line['dir'], page.rotation_matrix)
            else:
                polygon = quad_points(fitz.recover_line_quad(line), page)
            if not all(math.isfinite(v) for point in polygon for v in point):
                raise ValueError('Non-finite text coordinates')
            wmode = line.get('wmode')
            block_lines.append({'source_block': block_index, 'source_line_ids': [source_line_id], 'text': text, 'bbox': bounds(polygon),
                          'polygon': polygon, 'orientation': {0: 'horizontal', 1: 'vertical'}.get(wmode, 'unknown'),
                          'wmode': wmode, 'baseline': line['dir'], 'fonts': sorted(fonts), 'chars': chars})
        lines.extend(merge_vertical_fragments(block_lines))
    lines = merge_rotated_vertical_prefixes(lines, ~page.rotation_matrix)
    lines = merge_remaining_stacked_characters(lines)
    for number, line in enumerate(lines, 1):
        line['id'] = number
        for char in line['chars']:
            del char['origin']  # Used above in unrotated PDF coordinates only.
    all_text = ''.join(line['text'] for line in lines)
    bad = sorted(set(c for c in all_text if suspicious(c)))
    return {'page': page.number + 1, 'width': page.rect.width, 'height': page.rect.height,
            'rotation': page.rotation, 'coordinate_system': 'displayed_page_points_top_left',
            'char_count': sum(not c.isspace() for c in all_text),
            'suspicious_characters': [{'codepoint': f'U+{ord(c):04X}', 'count': all_text.count(c)} for c in bad],
            'image_count': len(page.get_image_info()), 'lines': lines}


def filter_document(fonts, pages, min_chars):
    reasons = []
    missing = [f for f in fonts if not f['to_unicode']]
    if missing:
        reasons.append({'code': 'font_without_to_unicode', 'fonts': missing})
    bad_pages = [{'page': p['page'], 'characters': p['suspicious_characters']}
                 for p in pages if p['suspicious_characters']]
    if bad_pages:
        reasons.append({'code': 'unreadable_character_codes', 'pages': bad_pages})
    total = sum(p['char_count'] for p in pages)
    if total < min_chars:
        reasons.append({'code': 'too_few_characters', 'actual': total, 'minimum': min_chars})
    if not fonts:
        reasons.append({'code': 'no_fonts'})
    return reasons


def render_page(page, data, target, dpi, prefix=None):
    image_name = f'{prefix}_page.png' if prefix else 'page.png'
    bbox_name = f'{prefix}_bbox.png' if prefix else 'bbox.png'
    labels_name = f'{prefix}_labels.json' if prefix else 'labels.json'
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csRGB, alpha=False)
    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    image.save(target / image_name)
    sx, sy = pix.width / data['width'], pix.height / data['height']
    # Union all strokes first, so intersecting boxes do not cancel each other.
    # Binary mask keeps the operation exactly RGB XOR 0xFFFFFF at marked pixels.
    mask = Image.new('1', image.size, 0)
    draw = ImageDraw.Draw(mask)
    font = ImageFont.load_default(size=max(12, round(dpi / 12)))
    stroke = max(2, round(dpi / 100))
    for line in data['lines']:
        pixel_polygon = [[x * sx, y * sy] for x, y in line['polygon']]
        pixel_bbox = bounds(pixel_polygon)
        line['polygon_pixels'] = pixel_polygon
        line['bbox_pixels'] = pixel_bbox
        draw.rectangle(pixel_bbox, outline=1, width=stroke)
        x, y = max(0, pixel_bbox[0]), max(0, pixel_bbox[1] - font.size - 2)
        label = str(line['id'])
        draw.text((x, y), label, font=font, fill=1)
    Image.composite(ImageOps.invert(image), image, mask).save(target / bbox_name)
    data.update(image_width=pix.width, image_height=pix.height, dpi=dpi,
                image=image_name, annotated_image=bbox_name,
                label_status='candidate_needs_visual_review',
                line_grouping='PDF lines, vertical fragment/rotated-prefix merging, then isolated-character vertical inference; no reading-order inference',
                geometry_method='embedded vertical glyph outlines when matched; PDF font metrics otherwise',
                bbox_edge_whitespace='excluded; original text and chars preserved')
    write_json(target / labels_name, data)


def prepare(source, output, min_chars=100, dpi=150):
    if min_chars < 1 or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError('min_chars must be >= 1; dpi must be finite and > 0')
    if shutil.which('pdffonts') is None:
        raise ValueError('pdffonts is required (Poppler)')
    pdfs = sorted(source.resolve().glob('*.pdf'))
    if not pdfs:
        raise ValueError(f'No PDFs found in {source}')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'status': 'running',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source': str(source.resolve()), 'min_chars': min_chars, 'dpi': dpi,
                'font_policy': 'strict: every pdffonts font must have an explicit ToUnicode map',
                'versions': {'pymupdf': fitz.VersionBind}, 'documents': []}
    write_json(output / 'manifest.json', manifest)
    for path in pdfs:
        record = {'name': path.name, 'source': str(path), 'status': 'error'}
        directory = None
        try:
            record['sha256'] = sha256(path)
            fonts, font_warnings = inspect_fonts(path)
            record.update(fonts=fonts, font_warnings=font_warnings)
            with fitz.open(path) as pdf:
                if pdf.needs_pass:
                    raise ValueError('Password required')
                pages = [extract_page(page) for page in pdf]
                reasons = filter_document(fonts, pages, min_chars)
                record.update(pages=len(pdf), char_count=sum(p['char_count'] for p in pages),
                              page_char_counts=[p['char_count'] for p in pages], reasons=reasons)
                if reasons:
                    record['status'] = 'skipped'
                else:
                    directory = output / path.stem
                    directory.mkdir()
                    for data in pages:
                        target = directory / f'page-{data["page"]:04d}'
                        target.mkdir()
                        render_page(pdf[data['page'] - 1], data, target, dpi)
                    if sha256(path) != record['sha256']:
                        raise ValueError('Source PDF changed during processing')
                    record.update(status='passed', output=path.stem)
        except Exception as error:
            record.update(status='error', error=str(error))
            if directory and directory.exists():
                # Only this new run's incomplete document output is removed.
                shutil.rmtree(directory)
        manifest['documents'].append(record)
        write_json(output / 'manifest.json', manifest)
        detail = ', '.join(r['code'] for r in record.get('reasons', [])) or record.get('error', '')
        print(f'{record["status"]:7} {path.name} {detail}', flush=True)
    manifest['status'] = 'complete'
    manifest['counts'] = {status: sum(d['status'] == status for d in manifest['documents'])
                          for status in ('passed', 'skipped', 'error')}
    manifest['counts']['rendered_pages'] = sum(d['pages'] for d in manifest['documents'] if d['status'] == 'passed')
    write_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('~/mocrdown/tests/data'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--min-chars', type=int, default=100, help='minimum non-whitespace character count per PDF')
    parser.add_argument('--dpi', type=float, default=150)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(ROOT):
        parser.error('--output must be inside textdet/')
    try:
        result = prepare(args.source.expanduser(), output, args.min_chars, args.dpi)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result['counts'], ensure_ascii=False))
    print(output)
    if result['counts']['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
