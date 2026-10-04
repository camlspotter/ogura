"""Generate JSON table recipes and render aligned supervision on demand."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
from pathlib import Path
from functools import lru_cache

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

WORDS = ['区分', '項目', '実績', '備考', '件数', '割合', '合計', '受付', '相談',
         '調査結果', '年度末時点', '継続対応', '対象外', '未集計', '第一地区',
         '第二地区', '参考資料', '確認済', 'ABC', '2026年度']
HARD_TEXTS = ['上', '1', 'I', 'ー', 'L', 'U', 'R', '月', '日', '回',
              '月　日', '年　月　日', '第1回', 'L U R', 'コ', 'エ', 'ロ', 'ニ',
              'カ', 'キ', 'ト', 'ヒ', 'コスト', 'チェック', '□', '□ 有　□ 無']
CONTENT_PROFILE = 'hard-negatives-v1'
MODES = ('plain', 'column_merge', 'row_merge', 'mixed', 'block_merge')
DEFAULT_SCALE = 4
SCALES = (2, 4, 8)


def sample_seed(seed: int, split: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}/{split}/{index}'.encode()).digest()[:8], 'big')


def partition(rows: int, columns: int, rng: random.Random, mode: str) -> list[dict]:
    """Tile the grid with disjoint rectangular cells; keep every track observable."""
    for _ in range(100):
        occupied = set()
        cells = []
        for r in range(rows):
            for c in range(columns):
                if (r, c) in occupied:
                    continue
                rs = cs = 1
                if mode != 'plain' and rng.random() < 0.35:
                    rs = rng.randint(1, min(3, rows-r)) if mode != 'column_merge' else 1
                    cs = rng.randint(1, min(3, columns-c)) if mode != 'row_merge' else 1
                    if mode == 'mixed' and rs > 1:
                        cs = 1
                    if any((y, x) in occupied for y in range(r, r+rs) for x in range(c, c+cs)):
                        rs = cs = 1
                occupied.update((y, x) for y in range(r, r+rs) for x in range(c, c+cs))
                cells.append(dict(row=r, column=c, rowspan=rs, colspan=cs))
        row_edges = {v for cell in cells for v in (cell['row'], cell['row']+cell['rowspan'])}
        col_edges = {v for cell in cells for v in (cell['column'], cell['column']+cell['colspan'])}
        has_required = (mode == 'plain' or
                        mode == 'column_merge' and any(c['colspan'] > 1 for c in cells) or
                        mode == 'row_merge' and any(c['rowspan'] > 1 for c in cells) or
                        mode == 'mixed' and any(c['rowspan'] > 1 for c in cells) and any(c['colspan'] > 1 for c in cells) or
                        mode == 'block_merge' and any(c['rowspan'] > 1 and c['colspan'] > 1 for c in cells))
        if len(row_edges) == rows+1 and len(col_edges) == columns+1 and has_required:
            return cells
    raise RuntimeError('Could not construct an observable table grid')


def boundary_segments(cells: list[dict], xs: list[float], ys: list[float]) -> tuple[list, list]:
    horizontal, vertical = set(), set()
    for cell in cells:
        r, c, rs, cs = (cell[k] for k in ('row', 'column', 'rowspan', 'colspan'))
        for y in (r, r+rs):
            horizontal.update((xs[x], ys[y], xs[x+1], ys[y]) for x in range(c, c+cs))
        for x in (c, c+cs):
            vertical.update((xs[x], ys[y], xs[x], ys[y+1]) for y in range(r, r+rs))
    return sorted(horizontal), sorted(vertical)


def draw_mask(size: tuple[int, int], segments: list, width: float, *,
              scale: int = 2, subpixel: bool = False) -> Image.Image:
    mask = Image.new('L', (size[0]*scale, size[1]*scale))
    draw = ImageDraw.Draw(mask)
    for segment in segments:
        if not subpixel:
            # Preserve schema-v2 rasterization exactly.
            draw.line(tuple(v*scale for v in segment), fill=255, width=int(width*scale))
            continue
        x0, y0, x1, y1 = segment
        if y0 == y1:
            bounds = (x0, y0-width/2, x1, y0+width/2)
        elif x0 == x1:
            bounds = (x0-width/2, y0, x0+width/2, y1)
        else:
            raise ValueError('Only horizontal/vertical segments are supported')
        # Pixel centers sample a half-open continuous rectangle. This avoids
        # Pillow's inclusive endpoint adding an extra pixel to the stroke width.
        left, top, right, bottom = [math.ceil(v*scale-0.5) for v in bounds]
        if right > left and bottom > top:
            draw.rectangle((left, top, right-1, bottom-1), fill=255)
    return mask


def wrap_text(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ''
    for char in text:
        if line and font.getlength(line+char) > width:
            lines.append(line)
            line = ''
        line += char
    if line:
        lines.append(line)
    return lines


def cell_content(rng: random.Random) -> tuple[str, str]:
    """Keep ordinary cells while oversampling line-like text and input-only marks."""
    choice = rng.random()
    if choice < 0.10:
        return 'diagonal', ''
    if choice < 0.20:
        return 'checkbox', ''
    if choice < 0.55:
        text = rng.choice(HARD_TEXTS + [f'△{rng.randint(1,99999):,}',
                                      f'▲{rng.randint(1,99999):,}'])
        return 'hard_text', text
    return 'ordinary', rng.choice(WORDS + [f'{rng.randint(0,99999):,}',
                                          f'{rng.uniform(0,100):.1f}%', '', ''])


def cell_marks(rng: random.Random, kind: str, bbox: list, foreground: str) -> list[dict]:
    """Resolve input-only marks in final-image coordinates; never table boundaries."""
    x0, y0, x1, y1 = bbox
    if kind == 'diagonal':
        inset = rng.choice([0, 0, 2, 4])
        a, b, c, d = x0+inset, y0+inset, x1-inset, y1-inset
        direction = rng.choice(['slash', 'backslash', 'cross'])
        segments = []
        if direction in ('slash', 'cross'):
            segments.append([a, d, c, b])
        if direction in ('backslash', 'cross'):
            segments.append([a, b, c, d])
        return [dict(kind='diagonal', segments=segments,
                     width=rng.choice([0.5, 0.75, 1, 1.5, 2]),
                     color=rng.choice([foreground, '#555555', '#888888']))]
    if kind == 'checkbox':
        side = rng.randint(9, 20)
        left = rng.uniform(x0+6, x1-6-side)
        top = rng.uniform(y0+6, y1-6-side)
        return [dict(kind='checkbox', bbox=[left, top, left+side, top+side],
                     width=rng.choice([0.75, 1, 1.5, 2]), color=foreground,
                     state=rng.choice(['empty', 'empty', 'checked', 'filled']))]
    return []


def make_sample(seed: int, font_path: Path, mode: str, *, font_record: dict | None = None, scale: int = DEFAULT_SCALE) -> dict:
    """Resolve all randomness and text layout without allocating any image."""
    if type(scale) is not int or scale not in SCALES:
        raise ValueError(f'scale must be one of {SCALES}')
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    font_record = font_record or validate_fonts([font_path])[0]
    rng = random.Random(seed)
    rows, columns = rng.randint(4, 16), rng.randint(3, 9)
    cells = partition(rows, columns, rng, mode)
    margin = rng.randint(12, 28)
    xs = [margin+rng.randrange(scale)/scale]
    ys = [margin+rng.randrange(scale)/scale]
    for _ in range(columns):
        xs.append(xs[-1]+rng.randint(75, 170)+rng.randrange(scale)/scale)
    for _ in range(rows):
        ys.append(ys[-1]+rng.randint(42, 80)+rng.randrange(scale)/scale)
    size = (math.ceil(xs[-1]+margin), math.ceil(ys[-1]+margin))
    inner_width = rng.choice([0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0, 3.0])
    outer_width = rng.choice([0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    h_segments, v_segments = boundary_segments(cells, xs, ys)
    header_color = rng.choice(['#e6edf5', '#eeeeee', '#e6efdf', '#ffffff', '#254b70'])
    stripe_color = rng.choice(['#ffffff', '#f1f4f7', '#f4f4ed'])
    font_size = rng.randint(12, 20)
    font = ImageFont.truetype(str(font_path), font_size*scale)
    ascent, descent = font.getmetrics()
    line_height = ascent+descent
    for index, cell in enumerate(cells):
        r, c, rs, cs = (cell[k] for k in ('row', 'column', 'rowspan', 'colspan'))
        bbox = [xs[c], ys[r], xs[c+cs], ys[r+rs]]
        x0, y0, x1, y1 = (value*scale for value in bbox)
        bg = header_color if r == 0 else stripe_color if r % 2 else '#ffffff'
        content_kind, text = cell_content(rng)
        # Keep the exact rendered text as the cell label; fit without clipping glyphs.
        cell_font, cell_height = font, line_height
        padding = (max(inner_width, outer_width)+4)*scale
        lines = wrap_text(text, cell_font, x1-x0-2*padding)
        cell_size = font_size
        while lines and (len(lines)*cell_height > y1-y0-2*padding or
                         max(cell_font.getlength(line) for line in lines) > x1-x0-2*padding):
            cell_size -= 1
            if cell_size < 6:
                raise RuntimeError('Cell text cannot fit')
            cell_font = ImageFont.truetype(str(font_path), cell_size*scale)
            cell_height = sum(cell_font.getmetrics())
            lines = wrap_text(text, cell_font, x1-x0-2*padding)
        align = rng.choice(['left', 'center', 'right'])
        top = y0+(y1-y0-len(lines)*cell_height)/2
        text_runs = []
        for offset, line in enumerate(lines):
            length = cell_font.getlength(line)
            left = x0+padding if align == 'left' else x1-padding-length if align == 'right' else (x0+x1-length)/2
            text_runs.append(dict(text=line, xy=[left/scale, (top+offset*cell_height)/scale]))
        cell.update(id=index, bbox=bbox, text=text, lines=lines, align=align, font_size=cell_size,
                    background=bg, foreground="white" if bg == "#254b70" else "#202020", text_runs=text_runs)
        cell['content_kind'] = content_kind
        cell['marks'] = cell_marks(rng, content_kind, bbox, cell['foreground'])
    line_color = rng.choice(['#111111', '#555555', '#888888', '#345778'])
    degradation = rng.choice(['clean', 'clean', 'jpeg', 'blur', 'downsample'])
    params = {'kind': degradation}
    if degradation == 'jpeg':
        params['quality'] = rng.randint(40, 90)
    elif degradation == 'blur':
        params['radius'] = rng.uniform(0.25, 0.7)
    elif degradation == 'downsample':
        params['factor'] = rng.uniform(0.5, 0.8)
    labels = dict(schema_version=4, renderer="pillow-table-v4", content_profile=CONTENT_PROFILE, scale=scale, seed=seed, mode=mode, width=size[0], height=size[1],
                  table_bbox=[xs[0], ys[0], xs[-1], ys[-1]], rows=rows, columns=columns,
                  x_boundaries=xs, y_boundaries=ys, cells=cells,
                  horizontal_segments=h_segments, vertical_segments=v_segments,
                  inner_line_width=inner_width, outer_line_width=outer_width,
                  line_color=line_color, header_color=header_color, stripe_color=stripe_color,
                  font=font_record, degradation=params)
    return labels



@lru_cache(maxsize=32)
def _font_digest(path: str, size: int, mtime_ns: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_sample(recipe: dict, *, font_dir: Path | None = None) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Render a JSON recipe in memory. No randomness, file output, or recipe mutation."""
    version = recipe.get('schema_version')
    scale = recipe.get('scale')
    if (type(scale) is not int or
            not ((version == 2 and recipe.get('renderer') == 'pillow-table-v2' and scale == 2) or
                 (version in (3, 4) and recipe.get('renderer') == f'pillow-table-v{version}' and scale in SCALES))):
        raise ValueError('Unsupported table recipe schema/renderer/scale')
    record = recipe['font']
    font_path = (Path(font_dir)/record['name'] if font_dir is not None else Path(record['path'])).resolve()
    stat = font_path.stat()
    if _font_digest(str(font_path), stat.st_size, stat.st_mtime_ns) != record['sha256']:
        raise ValueError(f'Font content differs from recipe: {font_path}')
    size = (recipe['width'], recipe['height'])
    xs, ys = recipe['x_boundaries'], recipe['y_boundaries']
    h_segments, v_segments = boundary_segments(recipe['cells'], xs, ys)
    inner_width, outer_width = recipe['inner_line_width'], recipe['outer_line_width']
    mask_options = dict(scale=scale, subpixel=version >= 3)
    # Outer frame uses its own width; thinner outer borders remain thin.
    h_inner = [s for s in h_segments if s[1] not in (ys[0], ys[-1])]
    v_inner = [s for s in v_segments if s[0] not in (xs[0], xs[-1])]
    h = ImageChops.lighter(draw_mask(size, h_inner, inner_width, **mask_options), draw_mask(size, [(xs[0], y, xs[-1], y) for y in (ys[0], ys[-1])], outer_width, **mask_options))
    v = ImageChops.lighter(draw_mask(size, v_inner, inner_width, **mask_options), draw_mask(size, [(x, ys[0], x, ys[-1]) for x in (xs[0], xs[-1])], outer_width, **mask_options))
    image = Image.new('RGB', h.size, 'white')
    draw = ImageDraw.Draw(image)
    fonts = {}
    for cell in recipe['cells']:
        draw.rectangle(tuple(v*scale for v in cell['bbox']), fill=cell['background'])
        fs = cell['font_size']
        if fs not in fonts:
            fonts[fs] = ImageFont.truetype(str(font_path), fs*scale)
        for run in cell['text_runs']:
            draw.text(tuple(v*scale for v in run['xy']), run['text'], font=fonts[fs],
                      anchor='lt', fill=cell['foreground'])
        for mark in cell.get('marks', []) if version >= 4 else []:
            width = max(1, round(mark['width']*scale))
            if mark['kind'] == 'diagonal':
                for segment in mark['segments']:
                    draw.line(tuple(value*scale for value in segment), fill=mark['color'], width=width)
            elif mark['kind'] == 'checkbox':
                box = tuple(value*scale for value in mark['bbox'])
                draw.rectangle(box, outline=mark['color'], width=width,
                               fill=mark['color'] if mark['state'] == 'filled' else None)
                if mark['state'] == 'checked':
                    left, top, right, bottom = box
                    side = right-left
                    draw.line([(left+0.2*side, top+0.5*side),
                               (left+0.45*side, top+0.8*side),
                               (left+0.85*side, top+0.2*side)], fill=mark['color'], width=width)
            else:
                raise ValueError(f"Unknown cell mark: {mark['kind']}")
    image.paste(recipe['line_color'], mask=ImageChops.lighter(h, v))
    image = image.resize(size, Image.Resampling.LANCZOS)
    h, v = (mask.resize(size, Image.Resampling.BOX) for mask in (h, v))
    params = recipe['degradation']
    if params['kind'] == 'jpeg':
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=params['quality'])
        buf.seek(0)
        with Image.open(buf) as decoded:
            image = decoded.convert('RGB')
    elif params['kind'] == 'blur':
        image = image.filter(ImageFilter.GaussianBlur(params['radius']))
    elif params['kind'] == 'downsample':
        small = tuple(max(1, round(s*params['factor'])) for s in size)
        image = image.resize(small, Image.Resampling.LANCZOS).resize(size, Image.Resampling.BICUBIC)
    elif params['kind'] != 'clean':
        raise ValueError(f"Unknown degradation: {params['kind']}")
    return image, h, v


class TableCellDataset:
    """Map-style dataset returning PIL images, compatible with custom DataLoader collation."""

    def __init__(self, root: Path, *, font_dir: Path | None = None):
        self.root = Path(root)
        self.font_dir = font_dir
        manifest = json.loads((self.root/'manifest.json').read_text())
        if manifest.get('schema_version') not in (2, 3, 4) or manifest.get('status') != 'complete':
            raise ValueError('Dataset must be a completed schema-v2/v3/v4 JSON dataset')
        self.records = [json.loads(line) for line in (self.root/'samples.jsonl').read_text().splitlines()]
        if len(self.records) != manifest['count']:
            raise ValueError('Sample count does not match manifest')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        recipe = json.loads((self.root/record['recipe']).read_text())
        image, horizontal, vertical = render_sample(recipe, font_dir=self.font_dir)
        return dict(id=record['id'], image=image, horizontal=horizontal, vertical=vertical, recipe=recipe)


def validate_fonts(paths: list[Path]) -> list[dict]:
    required = set(''.join(WORDS+HARD_TEXTS)+'0123456789,.%△▲') - {' ', '　'}
    result = []
    for path in paths:
        with TTFont(path) as font:
            cmap = font.getBestCmap() or {}
            missing = sorted(c for c in required if ord(c) not in cmap)
        if missing:
            raise ValueError(f'{path}: missing characters {missing}')
        result.append(dict(name=path.name, path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def generate(output: Path, count: int, seed: int, split: str, fonts: list[Path], *, scale: int = DEFAULT_SCALE) -> dict:
    if type(scale) is not int or scale not in SCALES:
        raise ValueError(f'scale must be one of {SCALES}')
    if count <= 0:
        raise ValueError('count must be positive')
    if not fonts:
        raise ValueError('At least one font is required')
    font_records = validate_fonts(fonts)
    output.mkdir(parents=True, exist_ok=False)
    (output/'recipes').mkdir()
    counts = dict.fromkeys(MODES, 0)
    manifest = dict(schema_version=4, content_profile=CONTENT_PROFILE, scale=scale, storage='json-only', status='generating', count=count, seed=seed, split=split,
                    fonts=font_records, pillow_version=Image.__version__,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mask_encoding='uint8 coverage, divide by 255; independent horizontal and vertical channels')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    with (output/'samples.jsonl').open('w') as records:
        for index in range(count):
            item_seed = sample_seed(seed, split, index)
            mode = MODES[index % len(MODES)]
            font_index = random.Random(item_seed).randrange(len(fonts))
            recipe = make_sample(item_seed, fonts[font_index], mode, font_record=font_records[font_index], scale=scale)
            name = f'table-{index:06d}'
            recipe_path = f'recipes/{name}.json'
            (output/recipe_path).write_text(json.dumps(recipe, ensure_ascii=False, indent=2)+'\n')
            records.write(json.dumps(dict(id=name, seed=item_seed, mode=mode, recipe=recipe_path))+'\n')
            counts[mode] += 1
            if (index+1) % 100 == 0:
                print(f'{split}: {index+1}/{count}', flush=True)
    manifest.update(status='complete', modes=counts)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=int, choices=SCALES, default=DEFAULT_SCALE,
                        help='Supersampling multiplier (default: 4)')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--font', type=Path, action='append', help='Repeat for multiple Japanese fonts')
    args = parser.parse_args()
    fonts = args.font or [Path(__file__).resolve().parents[3]/'corpus/fonts'/name for name in
                         ('NotoSansCJKjp-Regular.otf', 'NotoSerifCJKjp-Regular.otf')]
    try:
        result = generate(args.output, args.count, args.seed, args.split, [f.resolve() for f in fonts], scale=args.scale)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
