"""Generate JSON table recipes and render aligned supervision on demand."""
from __future__ import annotations

import argparse
import copy
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
HARD_TEXTS = ['し', '一', '上', '1', 'I', 'ー', 'L', 'U', 'R', '月', '日', '回',
              '月　日', '年　月　日', '第1回', 'L U R', 'コ', 'エ', 'ロ', 'ニ',
              'ユーザーリスト', 'ユーザー', 'リスト', 'ユーザー名',
              'カ', 'キ', 'ト', 'ヒ', 'コスト', 'チェック', '□', '□ 有　□ 無']
CONTENT_PROFILE = 'natural-forms-v5'
CHECK_ITEMS = ['ユーザーリストを確認した', 'チェック項目を確認した', '対象の区分を確認した',
               '参考資料を確認した', '調査結果を確認した', '月次の実績を確認した',
               '申請内容に変更はない', '担当者の確認が完了した', '記入漏れはない',
               '提出日を確認した', '金額を確認した', '回数を確認した',
               '受付番号を確認した', '連絡先を確認した', '備考欄を確認した',
               '添付資料を確認した', '承認が完了した', '対象人数を確認した',
               '年度の区分を確認した', '実施日を確認した', '使用目的を確認した']
LEDGER_ITEMS = ['現金', '預金', '売上', '費用', '備品', '消耗品', '通信費', '旅費',
                '手数料', '使用料', '図書費', '修繕費', '給与', '雑費', 'その他']
FORM_ITEMS = ['氏名', '所属', '住所', '連絡先', '受付日', '申請日', '区分', '担当者',
              '部署', '備考', '項目', '対象', '期間', '件数', '回数', '確認', '承認',
              '年度', '支払日', '金額', '使用目的', '資料名', '受付番号', '確認日',
              '実施日', '担当部署', '対象人数', '結果', '作成日', '提出日', '更新日', '摘要',
              '資料番号', '管理番号', '処理日', '備考欄']
TEMPLATE_TEXTS = CHECK_ITEMS+LEDGER_ITEMS+FORM_ITEMS+['番号', '確認項目', 'はい', 'いいえ',
    '科目', '前年度', '当年度', '増減', '小計', '合計', '項目', '内容', '担当', '備考']
EMPTY_SIGNS = ['ー', '―', '−', '－', '—']
CELL_COLORS = ['#ffcccc', '#f59ac7', '#fbd5b5', '#ccecff', '#ccffcc', '#fff2cc']
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
    if choice < 0.28:
        return 'empty_sign', rng.choice(EMPTY_SIGNS)
    if choice < 0.36:
        return 'arrow', ''
    if choice < 0.65:
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
        side = min(rng.randint(9, 20), y1-y0-4, x1-x0-4)
        left = rng.uniform(x0+2, x1-2-side)
        top = rng.uniform(y0+2, y1-2-side)
        return [dict(kind='checkbox', bbox=[left, top, left+side, top+side],
                     width=rng.choice([0.75, 1, 1.5, 2]), color=foreground,
                     state=rng.choice(['empty', 'empty', 'checked', 'filled']))]
    if kind == 'arrow':
        y = (y0+y1)/2
        start, end = [x0+6, y], [x1-6, y]
        if rng.random() < 0.25:
            start, end = [x0+6, y0+4], [x1-6, y1-4]
        return [dict(kind='arrow', start=start, end=end,
                     head=min(rng.uniform(3, 9), (y1-y0)/3),
                     double=rng.random() < 0.4, width=rng.choice([0.5, 1, 1.5, 2]), color=foreground)]
    return []


def dashed_segments(segments: list, styles: dict) -> list:
    """Split input strokes with a phase shared across every piece of a boundary."""
    result = []
    for x0, y0, x1, y1 in segments:
        horizontal = y0 == y1
        style = styles.get(str(y0 if horizontal else x0))
        if style is None:
            result.append((x0, y0, x1, y1))
            continue
        dash, gap, phase = (style[k] for k in ('dash', 'gap', 'phase'))
        if dash <= 0 or gap <= 0:
            raise ValueError('Dash and gap must be positive')
        start, end = (x0, x1) if horizontal else (y0, y1)
        period = dash+gap
        position = math.floor((start-phase)/period)*period+phase
        while position < end:
            left, right = max(start, position), min(end, position+dash)
            if right > left:
                result.append((left, y0, right, y1) if horizontal else (x0, left, x1, right))
            position += period
    return result


def line_styles(rng, xs, ys):
    styles = {'horizontal': {}, 'vertical': {}}
    # Half of tables mix solid and dashed internal boundaries; frames stay solid.
    if rng.random() < 0.5:
        for channel, coordinates in [('horizontal', ys), ('vertical', xs)]:
            probability = 0.6 if channel == 'horizontal' else 0.25
            for coordinate in coordinates[1:-1]:
                if rng.random() < probability:
                    styles[channel][str(coordinate)] = dict(
                        dash=rng.choice([1, 2, 3, 5, 8]),
                        gap=rng.choice([1, 1.5, 2, 3, 5]), phase=rng.uniform(0, 4))
    return styles


def template_content(rng, template, row, column, columns, rows, colspan, context):
    """Assign roles by position instead of independently scattering symbols."""
    if template == 'checklist':
        if row == 0:
            return 'ordinary', ['番号', '確認項目', 'はい', 'いいえ'][column], 'center'
        if column == 0 and colspan == 1:
            return 'ordinary', str(row), 'center'
        if column <= 1 or colspan > 1:
            return 'hard_text', context['check_items'][row-1], 'left'
        return 'checkbox', '', 'center'
    if template == 'ledger':
        if row == 0:
            return 'ordinary', ['科目', '前年度', '当年度', '増減'][column], 'center'
        record = context['ledger'][row-1]
        if column == 0 or colspan > 1:
            return 'ordinary', record['label'], 'left'
        value = record['values'][column-1]
        if value == 0:
            return 'empty_sign', '―', 'center'
        return 'ordinary', ('△' if value < 0 else '')+f'{abs(value):,}', 'right'
    if row == 0:
        return 'ordinary', ['項目','内容'][column % 2], 'center'
    field = FORM_ITEMS[((row-1)*(columns//2)+column//2) % len(FORM_ITEMS)]
    if column % 2 == 0:
        return 'ordinary', field, 'left'
    if field.endswith('日') or field == '期間':
        return 'ordinary', '月　日', 'center'
    if field in ('区分','確認','承認'):
        return 'checkbox', '', 'center'
    if rng.random() < 0.08:
        return 'diagonal', '', 'center'
    if field == '備考' and rng.random() < 0.15:
        return 'arrow', '', 'center'
    return 'ordinary', '', 'left'


def template_context(rng, rows):
    items = rng.sample(CHECK_ITEMS, len(CHECK_ITEMS))
    records, group, all_values = [], [], []
    for row in range(1, rows):
        if row == rows-1 or row % 6 == 0:
            source = all_values if row == rows-1 else group
            values = [sum(v[c] for v in source) for c in range(3)]
            label = '合計' if row == rows-1 else '小計'
            group = []
        else:
            previous, current = [0 if rng.random() < 0.15 else rng.randint(1,999)*100 for _ in range(2)]
            values = [previous, current, current-previous]
            label = LEDGER_ITEMS[len(all_values) % len(LEDGER_ITEMS)]
            group.append(values)
            all_values.append(values)
        records.append(dict(label=label, values=values))
    return dict(check_items=items, ledger=records,
                answers=[rng.choices(['yes','no','blank'],weights=[5,4,1])[0] for _ in range(rows)],
                checkbox_width=rng.choice([0.75,1,1.25]))


def make_sample(seed: int, font_path: Path, mode: str, *, font_record: dict | None = None, scale: int = DEFAULT_SCALE) -> dict:
    """Resolve all randomness and text layout without allocating any image."""
    if type(scale) is not int or scale not in SCALES:
        raise ValueError(f'scale must be one of {SCALES}')
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    font_record = font_record or validate_fonts([font_path])[0]
    rng = random.Random(seed)
    template = rng.choices(['checklist', 'ledger', 'form', 'stress'], weights=[40,35,20,5])[0]
    compact = template in ('checklist', 'ledger') or template == 'stress' and rng.random() < 0.4
    rows, columns = rng.randint(12, 28) if compact else rng.randint(4, 16), rng.randint(3, 9)
    if template != 'stress':
        columns = 4 if template != 'form' else rng.choice([4, 6])
        rows = rng.randint(10, 22) if compact else rng.randint(6, 12)
    wide_column = 1 if template == 'checklist' else 0 if compact else None
    cells = partition(rows, columns, rng, mode)
    margin = rng.randint(12, 28)
    xs = [margin+rng.randrange(scale)/scale]
    ys = [margin+rng.randrange(scale)/scale]
    for column in range(columns):
        xs.append(xs[-1]+(rng.randint(250, 450) if len(xs)-1 == wide_column else rng.randint(75, 170))+rng.randrange(scale)/scale)
    if template != 'stress':
        widths = ([rng.randint(30, 45), rng.randint(300, 450), rng.randint(55, 75), rng.randint(55, 75)]
                  if template == 'checklist' else
                  [rng.randint(200, 300)]+[rng.randint(100, 145) for _ in range(columns-1)] if template == 'ledger' else
                  [rng.randint(80, 110) if c % 2 == 0 else rng.randint(140, 200) for c in range(columns)])
        xs = [xs[0]]
        for width in widths:
            xs.append(xs[-1]+width+rng.randrange(scale)/scale)
    for row in range(rows):
        ys.append(ys[-1]+(rng.randint(14, 30) if compact and rng.random() < 0.8 else rng.randint(42, 80))+rng.randrange(scale)/scale)
    if template != 'stress':
        ys = [ys[0]]
        row_height = rng.randint(20, 28) if compact else rng.randint(38, 55)
        if compact and rng.random() < 0.25:
            row_height = rng.randint(14, 18)
        for row in range(rows):
            height = row_height
            if row == 0:
                height = rng.randint(28, 40)
            ys.append(ys[-1]+height+rng.randrange(scale)/scale)
    size = (math.ceil(xs[-1]+margin), math.ceil(ys[-1]+margin))
    inner_width = rng.choice([0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0, 3.0])
    outer_width = rng.choice([0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    h_segments, v_segments = boundary_segments(cells, xs, ys)
    header_color = rng.choice(['#e6edf5', '#eeeeee', '#e6efdf', '#ffffff', '#254b70'])
    stripe_color = rng.choice(['#ffffff', '#f1f4f7', '#f4f4ed'])
    background_mode = rng.choice(['legacy', 'rows', 'columns', 'cells'])
    row_colors = [rng.choice(CELL_COLORS+['#ffffff']*3) for _ in range(rows)]
    column_colors = [rng.choice(CELL_COLORS+['#ffffff']*3) for _ in range(columns)]
    if template != 'stress':
        inner_width = rng.choice([0.5, 0.75, 1, 1.5])
        outer_width = rng.choice([1, 1.5, 2])
    accent = rng.choice(CELL_COLORS)
    font_size = min(rng.randint(12,16), max(8, round(row_height/1.8))) if template != 'stress' and compact else rng.randint(12, 18)
    context = template_context(rng, rows)
    font_cache = {}
    def sized_font(size):
        if size not in font_cache:
            font_cache[size] = ImageFont.truetype(str(font_path), size*scale)
        return font_cache[size]
    font = sized_font(font_size)
    ascent, descent = font.getmetrics()
    line_height = ascent+descent
    for index, cell in enumerate(cells):
        r, c, rs, cs = (cell[k] for k in ('row', 'column', 'rowspan', 'colspan'))
        bbox = [xs[c], ys[r], xs[c+cs], ys[r+rs]]
        x0, y0, x1, y1 = (value*scale for value in bbox)
        bg = header_color if r == 0 else stripe_color if r % 2 else '#ffffff'
        if background_mode == 'rows':
            bg = row_colors[r]
        elif background_mode == 'columns':
            bg = column_colors[c]
        elif background_mode == 'cells':
            bg = rng.choice(CELL_COLORS+['#ffffff']*3)
        content_kind, text = cell_content(rng)
        if bbox[3]-bbox[1] < 32 and content_kind in ('ordinary', 'hard_text'):
            text = rng.choice(WORDS+HARD_TEXTS)

        preferred_align = None
        if template != 'stress':
            content_kind, text, preferred_align = template_content(rng, template, r, c, columns, rows, cs, context)
            bg = header_color if r == 0 else '#ffffff'
            if template == 'checklist' and r > 0 and c == 2 and cs == 1:
                bg = accent
            if template == 'ledger' and (r == rows-1 or r % 6 == 0):
                bg = accent
        # Keep the exact rendered text as the cell label; fit without clipping glyphs.
        cell_font, cell_height = font, line_height
        if content_kind == 'empty_sign' and template == 'stress':
            cell_font = sized_font(rng.randint(10, 28))
            cell_height = sum(cell_font.getmetrics())
        padding = min(max(inner_width, outer_width)+4, max(1, (bbox[3]-bbox[1]-10)/2))*scale
        if compact and template != 'stress':
            padding = 3*scale if row_height >= 20 else scale
        lines = wrap_text(text, cell_font, x1-x0-2*padding)
        cell_size = cell_font.size//scale
        while lines and (len(lines)*cell_height > y1-y0-2*padding or
                         max(cell_font.getlength(line) for line in lines) > x1-x0-2*padding):
            cell_size -= 1
            if cell_size < 6:
                raise RuntimeError('Cell text cannot fit')
            cell_font = sized_font(cell_size)
            cell_height = sum(cell_font.getmetrics())
            lines = wrap_text(text, cell_font, x1-x0-2*padding)
        align = preferred_align or ('center' if content_kind == 'empty_sign' else rng.choice(['left', 'center', 'right']))
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
        cell['text_stroke_width'] = 0.25 if template != 'stress' and (r == 0 or text in ('小計','合計')) else 0
        if template == 'form' and c % 2 == 0 and r > 0:
            cell['background'] = '#f2f2f2'
        if template == 'checklist' and content_kind == 'checkbox':
            for mark in cell['marks']:
                side = min(12, bbox[3]-bbox[1]-6)
                left, top = (bbox[0]+bbox[2]-side)/2, (bbox[1]+bbox[3]-side)/2
                mark['bbox'] = [left,top,left+side,top+side]
                mark['state'] = 'checked' if context['answers'][r] == ('yes' if c == 2 else 'no') else 'empty'
                mark['width'] = context['checkbox_width']
        # A background change inside a cell is not a structural boundary.
        if template == 'stress' and rng.random() < 0.1:
            cell['background_patch'] = dict(bbox=[bbox[0], bbox[1], (bbox[0]+bbox[2])/2, bbox[3]],
                                            color=rng.choice(CELL_COLORS))
    line_color = rng.choice(['#111111', '#333333', '#555555']) if template != 'stress' else rng.choice(['#111111', '#555555', '#888888', '#345778'])
    degradation = rng.choice(['clean', 'clean', 'jpeg', 'blur', 'downsample'])
    params = {'kind': degradation}
    if degradation == 'jpeg':
        params['quality'] = rng.randint(40, 90)
    elif degradation == 'blur':
        params['radius'] = rng.uniform(0.25, 0.7)
    elif degradation == 'downsample':
        params['factor'] = rng.uniform(0.5, 0.8)
    labels = dict(schema_version=8, renderer="pillow-table-v8", content_profile=CONTENT_PROFILE, scale=scale, seed=seed, mode=mode, width=size[0], height=size[1],
                  table_bbox=[xs[0], ys[0], xs[-1], ys[-1]], rows=rows, columns=columns,
                  x_boundaries=xs, y_boundaries=ys, cells=cells,
                  horizontal_segments=h_segments, vertical_segments=v_segments,
                  inner_line_width=inner_width, outer_line_width=outer_width,
                  line_color=line_color, header_color=header_color, stripe_color=stripe_color,
                  font=font_record, degradation=params)
    labels['line_styles'] = line_styles(rng, xs, ys)
    labels['geometry_profile'] = 'compact' if compact else 'regular'
    labels['background_mode'] = background_mode
    labels['corner_radius'] = min(rng.uniform(3, 12), (ys[1]-ys[0])/3,
                                  (ys[-1]-ys[-2])/3, (xs[1]-xs[0])/3,
                                  (xs[-1]-xs[-2])/3) if rng.random() < 0.3 else 0
    labels['boundary_appearance'] = {'horizontal': {}, 'vertical': {}}
    for channel, coordinates in [('horizontal', ys), ('vertical', xs)]:
        for coordinate in coordinates[1:-1]:
            if rng.random() < 0.3:
                labels['boundary_appearance'][channel][str(coordinate)] = dict(
                    color=rng.choice(['#111111', '#ff0000', '#2469a0', '#777777']),
                    width=rng.choice([0.5, 0.75, 1, 1.5, 2, 3]))
    labels['template'] = template
    if template == 'ledger':
        labels['ledger_records'] = context['ledger']
    if template != 'stress':
        labels['background_mode'] = 'answer-column' if template == 'checklist' else 'subtotal-rows' if template == 'ledger' else 'header'
        labels['boundary_appearance'] = {'horizontal': {}, 'vertical': {}}
        labels['line_styles'] = {'horizontal': {}, 'vertical': {}}
        labels['corner_radius'] = min(10, (ys[1]-ys[0])/3) if template == 'form' and rng.random() < 0.7 else 0
        if template in ('checklist','form') and rng.random() < 0.5:
            dash, gap = rng.choice([1,2,4]), rng.choice([1,2,3])
            for coordinate in ys[2:-1:3]:
                labels['line_styles']['horizontal'][str(coordinate)] = dict(dash=dash,gap=gap,phase=0)
        if template == 'ledger' and rng.random() < 0.3:
            coordinate = xs[-2]
            labels['boundary_appearance']['vertical'][str(coordinate)] = dict(color='#e00000',width=2)
    return labels



@lru_cache(maxsize=32)
def _font_digest(path: str, size: int, mtime_ns: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stroke_groups(segments, recipe, channel):
    groups = {}
    for segment in segments:
        coordinate = segment[1] if channel == 'horizontal' else segment[0]
        appearance = recipe['boundary_appearance'][channel].get(str(coordinate), {})
        key = (appearance.get('color', recipe['line_color']),
               appearance.get('width', recipe['inner_line_width']))
        groups.setdefault(key, []).append(segment)
    return groups


def styled_mask(size, segments, recipe, channel, scale):
    mask = Image.new('L', (size[0]*scale, size[1]*scale))
    for (_, width), group in stroke_groups(segments, recipe, channel).items():
        mask = ImageChops.lighter(mask, draw_mask(size, group, width, scale=scale, subpixel=True))
    return mask


def _render_structural_sample(recipe: dict, *, font_dir: Path | None = None) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Render a JSON recipe in memory. No randomness, file output, or recipe mutation."""
    version = recipe.get('schema_version')
    scale = recipe.get('scale')
    if (type(scale) is not int or
            not ((version == 2 and recipe.get('renderer') == 'pillow-table-v2' and scale == 2) or
                 (version in (3, 4, 5, 6, 7, 8) and recipe.get('renderer') == f'pillow-table-v{version}' and scale in SCALES))):
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
        patch = cell.get('background_patch') if version >= 6 else None
        if patch:
            draw.rectangle(tuple(v*scale for v in patch['bbox']), fill=patch['color'])
        fs = cell['font_size']
        if fs not in fonts:
            fonts[fs] = ImageFont.truetype(str(font_path), fs*scale)
        for run in cell['text_runs']:
            draw.text(tuple(v*scale for v in run['xy']), run['text'], font=fonts[fs],
                      anchor='lt', fill=cell['foreground'],
                      stroke_width=round(cell.get('text_stroke_width', 0)*scale) if version >= 8 else 0,
                      stroke_fill=cell['foreground'])
        for mark in cell.get('marks', []) if version >= 4 else []:
            width = max(1, round(mark['width']*scale))
            if mark['kind'] == 'diagonal':
                for segment in mark['segments']:
                    draw.line(tuple(value*scale for value in segment), fill=mark['color'], width=width)
            elif mark['kind'] == 'arrow':
                a, b = mark['start'], mark['end']
                draw.line(tuple(v*scale for point in (a, b) for v in point), fill=mark['color'], width=width)
                theta = math.atan2(b[1]-a[1], b[0]-a[0])
                for tip, angle in [(b, theta)]+([(a, theta+math.pi)] if mark['double'] else []):
                    points = [(tip[0]-mark['head']*math.cos(angle+delta),
                               tip[1]-mark['head']*math.sin(angle+delta)) for delta in (-0.5, 0.5)]
                    draw.line([tuple(v*scale for v in points[0]), tuple(v*scale for v in tip),
                               tuple(v*scale for v in points[1])], fill=mark['color'], width=width)
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
    input_mask = ImageChops.lighter(h, v)
    if version >= 5:
        styles = recipe['line_styles']
        input_h = draw_mask(size, dashed_segments(h_inner, styles['horizontal']), inner_width, **mask_options)
        input_v = draw_mask(size, dashed_segments(v_inner, styles['vertical']), inner_width, **mask_options)
        frame_h = draw_mask(size, [(xs[0], y, xs[-1], y) for y in (ys[0], ys[-1])], outer_width, **mask_options)
        frame_v = draw_mask(size, [(x, ys[0], x, ys[-1]) for x in (xs[0], xs[-1])], outer_width, **mask_options)
        input_mask = ImageChops.lighter(ImageChops.lighter(input_h, input_v), ImageChops.lighter(frame_h, frame_v))
    if version >= 6:
        h = ImageChops.lighter(styled_mask(size, h_inner, recipe, 'horizontal', scale), frame_h)
        v = ImageChops.lighter(styled_mask(size, v_inner, recipe, 'vertical', scale), frame_v)
        for channel, segments in [('horizontal', h_inner), ('vertical', v_inner)]:
            for (color, width), group in stroke_groups(segments, recipe, channel).items():
                mask = draw_mask(size, dashed_segments(group, recipe['line_styles'][channel]), width, **mask_options)
                image.paste(color, mask=mask)
        radius = recipe['corner_radius']
        if radius:
            shape = Image.new('L', image.size)
            box = tuple(value*scale for value in recipe['table_bbox'])
            ImageDraw.Draw(shape).rounded_rectangle(box, radius=radius*scale, fill=255)
            image = Image.composite(image, Image.new('RGB', image.size, 'white'), shape)
            outline = Image.new('L', image.size)
            ImageDraw.Draw(outline).rounded_rectangle(box, radius=radius*scale, outline=255,
                                                     width=max(1, round(outer_width*scale)))
            image.paste(recipe['line_color'], mask=outline)
        else:
            image.paste(recipe['line_color'], mask=ImageChops.lighter(frame_h, frame_v))
    else:
        image.paste(recipe['line_color'], mask=input_mask)
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


def add_dense_cells(recipe, seed):
    """Resolve a paragraph-heavy form using only font metrics and JSON geometry."""
    recipe = copy.deepcopy(recipe)
    rng = random.Random(f'dense-cells/{seed}')
    scale = recipe['scale']
    rows = rng.randint(5, 9)
    xs = [20, 65, rng.randint(420, 560), rng.randint(620, 720)]
    ys = [20, 52]
    for _ in range(rows-1):
        ys.append(ys[-1]+rng.randint(65, 110))
    cells = partition(rows, 3, rng, recipe['mode'])
    for index, cell in enumerate(cells):
        r,c,rs,cs = (cell[k] for k in ('row','column','rowspan','colspan'))
        box = [xs[c],ys[r],xs[c+cs],ys[r+rs]]
        fs = rng.choice([10,11,12])
        font = ImageFont.truetype(recipe['font']['path'], fs*scale)
        pitch = sum(font.getmetrics())/scale
        width = (box[2]-box[0]-10)*scale
        if r == 0:
            text = ['番号','確認項目','結果'][c]
        elif box[2]-box[0] > 180:
            phrases = rng.sample(CHECK_ITEMS, 12)
            text = '。'.join(phrases)+'。'
        else:
            text = str(r) if c == 0 else rng.choice(HARD_TEXTS)
        lines = wrap_text(text,font,width)
        lines = lines[:max(1,int((box[3]-box[1]-10)/pitch))]
        # Store exactly what is rendered, with no hidden/truncated label text.
        runs = [dict(text=line,xy=[box[0]+5,box[1]+5+i*pitch]) for i,line in enumerate(lines)]
        cell.update(id=index,bbox=box,text='\n'.join(lines),lines=lines,align='left',
                    font_size=fs,background='#eeeeee' if r==0 else '#ffffff',
                    foreground='#202020',text_runs=runs,marks=[],content_kind='dense_text',
                    text_stroke_width=0)
    h,v = boundary_segments(cells,xs,ys)
    recipe.update(template='dense_form',content_profile='dense-paragraph-v1',rows=rows,columns=3,
                  width=xs[-1]+20,height=ys[-1]+20,table_bbox=[xs[0],ys[0],xs[-1],ys[-1]],
                  x_boundaries=xs,y_boundaries=ys,cells=cells,horizontal_segments=h,vertical_segments=v,
                  line_styles={'horizontal':{},'vertical':{}},
                  boundary_appearance={'horizontal':{},'vertical':{}},corner_radius=0)
    recipe.pop('ledger_records',None)
    return recipe


def add_dashed_borders(recipe, seed):
    """Resolve dots, short/long dashes and faint strokes into existing recipe fields."""
    recipe = copy.deepcopy(recipe)
    rng = random.Random(f'dashed-borders/{seed}')
    recipe['dash_profile'] = 'varied-dashes-v1'
    recipe['line_styles'] = {'horizontal':{},'vertical':{}}
    patterns = [(0.75,1.5),(1,2),(2,2),(3,3),(6,3),(10,5)]
    for channel, coordinates in [('horizontal',recipe['y_boundaries']),
                                 ('vertical',recipe['x_boundaries'])]:
        selected = [value for value in coordinates[1:-1]
                    if rng.random() < (0.55 if channel=='horizontal' else 0.25)]
        if channel=='horizontal' and not selected and len(coordinates)>2:
            selected = [rng.choice(coordinates[1:-1])]
        dash,gap = rng.choice(patterns)
        width = rng.choice([0.5,0.75,1,1.25])
        color = rng.choice(['#222222','#444444','#777777'])
        phase = rng.uniform(0,dash+gap)
        for value in selected:
            key = str(value)
            recipe['line_styles'][channel][key] = dict(dash=dash,gap=gap,phase=phase)
            recipe['boundary_appearance'][channel][key] = dict(width=width,color=color)
    return recipe


def add_content_lines(recipe, seed):
    """Resolve text underlines and shallow non-boundary diagonals into JSON marks."""
    recipe=copy.deepcopy(recipe)
    rng=random.Random(f'content-lines/{seed}')
    recipe['content_lines_profile']='underlines-empty-cell-diagonals-v2'
    fonts={}
    for cell in recipe['cells']:
        x0,y0,x1,y1=cell['bbox']
        if cell['row']==0: continue
        # An empty-cell cancellation mark connects actual corners; slope follows geometry.
        if (x1-x0)/(y1-y0)>=3 and rng.random()<.35:
            angle=math.degrees(math.atan2(y1-y0,x1-x0))
            cell.update(text='',lines=[],text_runs=[],content_kind='empty_diagonal',marks=[
                dict(kind='diagonal',role='shallow_diagonal',angle_degrees=angle,
                     segments=[[x0,y0,x1,y1]],width=rng.choice([.5,.75,1,1.5]),
                     color=cell['foreground'])])
            continue
        fs=cell['font_size']
        if fs not in fonts: fonts[fs]=ImageFont.truetype(recipe['font']['path'],fs*recipe['scale'])
        font=fonts[fs]
        for run in cell['text_runs']:
            if not run['text'].strip() or rng.random()>=.4: continue
            x,y=run['xy']
            _,_,_,bottom=font.getbbox(run['text'],anchor='lt')
            length=font.getlength(run['text'])/recipe['scale']
            # A phrase or whole line, placed just below the rendered glyphs.
            fraction=rng.choice([.3,.6,1.0])
            start=x+rng.uniform(0,max(0,length*(1-fraction)))
            end=min(x1-4,start+length*fraction)
            baseline=y+bottom/recipe['scale']+rng.choice([1,2,3])
            if end-start>=5 and baseline<y1-4:
                cell['marks'].append(dict(kind='diagonal',role='underline',
                    segments=[[start,baseline,end,baseline]],width=rng.choice([.5,.75,1,1.5]),
                    color=cell['foreground']))
    return recipe


def add_background_context(recipe, seed, *, negative_heavy=False):
    """Resolve page whitespace and zero-target negatives into a schema-v9 recipe."""
    recipe = copy.deepcopy(recipe)
    rng = random.Random(f'background-context/{seed}')
    kind = rng.choices(['table', 'page_table', 'text_only', 'blank'], weights=[40,20,30,10] if negative_heavy else [40,40,10,10])[0]
    recipe.update(schema_version=9, renderer='pillow-table-v9', background_context=kind, negative_heavy=negative_heavy)
    if kind == 'page_table':
        width, height = recipe['width'], recipe['height']
        extra_width = rng.randint(0, max(1, width//3))
        extra_height = rng.randint(max(1,height//3), max(1,height))
        dx = rng.randint(0,extra_width)
        dy = rng.randint(0, max(1,extra_height//3))
        recipe['width'] += extra_width
        recipe['height'] += extra_height
        def point(values):
            return [value+(dx if i%2==0 else dy) for i,value in enumerate(values)]
        recipe['table_bbox'] = point(recipe['table_bbox'])
        for channel,key,delta in [('horizontal','y_boundaries',dy),('vertical','x_boundaries',dx)]:
            recipe[key] = [value+delta for value in recipe[key]]
            for styles_key in ('line_styles','boundary_appearance'):
                recipe[styles_key][channel] = {str(float(coordinate)+delta):style
                    for coordinate,style in recipe[styles_key][channel].items()}
        for key in ('horizontal_segments','vertical_segments'):
            recipe[key] = [point(segment) for segment in recipe[key]]
        for cell in recipe['cells']:
            cell['bbox'] = point(cell['bbox'])
            for run in cell['text_runs']:
                run['xy'] = point(run['xy'])
            if 'background_patch' in cell:
                cell['background_patch']['bbox'] = point(cell['background_patch']['bbox'])
            for mark in cell['marks']:
                for key in ('bbox','start','end'):
                    if key in mark:
                        mark[key] = point(mark[key])
                if 'segments' in mark:
                    mark['segments'] = [point(segment) for segment in mark['segments']]
        recipe['page_offset'] = [dx,dy]
    return recipe




def render_sample(recipe, *, font_dir=None):
    if recipe.get('schema_version') != 9:
        return _render_structural_sample(recipe, font_dir=font_dir)
    if recipe.get('renderer') != 'pillow-table-v9' or recipe.get('background_context') not in ('table','page_table','text_only','blank'):
        raise ValueError('Unsupported table background context')
    size = (recipe['width'],recipe['height'])
    if type(recipe.get('scale')) is not int or recipe['scale'] not in SCALES:
        raise ValueError('Unsupported table recipe scale')
    if recipe['background_context']=='blank':
        return Image.new('RGB',size,'white'),Image.new('L',size),Image.new('L',size)
    base = copy.deepcopy(recipe)
    base.update(schema_version=8, renderer='pillow-table-v8')
    if recipe['background_context']=='text_only':
        base['line_color'] = '#ffffff'
        base['corner_radius'] = 0
        for channel in base['boundary_appearance'].values():
            for style in channel.values():
                style['color'] = '#ffffff'
        for cell in base['cells']:
            cell['background'] = '#ffffff'
            cell['foreground'] = '#202020'
            cell.pop('background_patch',None)
    image,h,v = _render_structural_sample(base,font_dir=font_dir)
    if recipe['background_context']=='text_only':
        h,v = Image.new('L',size),Image.new('L',size)
    return image,h,v


class TableCellDataset:
    """Map-style dataset returning PIL images, compatible with custom DataLoader collation."""

    def __init__(self, root: Path, *, font_dir: Path | None = None):
        self.root = Path(root)
        self.font_dir = font_dir
        manifest = json.loads((self.root/'manifest.json').read_text())
        if manifest.get('schema_version') not in (2, 3, 4, 5, 6, 7, 8, 9) or manifest.get('status') != 'complete':
            raise ValueError('Dataset must be a completed schema-v2/v3/v4/v5/v6/v7/v8/v9 JSON dataset')
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
    required = set(''.join(WORDS+HARD_TEXTS+EMPTY_SIGNS+TEMPLATE_TEXTS)+'0123456789,.%△▲') - {' ', '　'}
    result = []
    for path in paths:
        with TTFont(path) as font:
            cmap = font.getBestCmap() or {}
            missing = sorted(c for c in required if ord(c) not in cmap)
        if missing:
            raise ValueError(f'{path}: missing characters {missing}')
        result.append(dict(name=path.name, path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def generate(output: Path, count: int, seed: int, split: str, fonts: list[Path], *, scale: int = DEFAULT_SCALE, background_context: bool = False, dense_text: bool = False, dashed_borders: bool = False, negative_heavy: bool = False, content_lines: bool = False) -> dict:
    if type(scale) is not int or scale not in SCALES:
        raise ValueError(f'scale must be one of {SCALES}')
    if count <= 0:
        raise ValueError('count must be positive')
    if not fonts:
        raise ValueError('At least one font is required')
    if negative_heavy and not background_context:
        raise ValueError("negative_heavy requires background_context")
    font_records = validate_fonts(fonts)
    output.mkdir(parents=True, exist_ok=False)
    (output/'recipes').mkdir()
    counts = dict.fromkeys(MODES, 0)
    manifest = dict(schema_version=9 if background_context else 8, background_context=background_context, dense_text=dense_text, dashed_borders=dashed_borders, negative_heavy=negative_heavy, content_lines=content_lines, content_profile=CONTENT_PROFILE, scale=scale, storage='json-only', status='generating', count=count, seed=seed, split=split,
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
            if dense_text and random.Random(f'dense-selection/{item_seed}').random() < 0.3:
                recipe = add_dense_cells(recipe, item_seed)
            if dashed_borders:
                recipe = add_dashed_borders(recipe, item_seed)
            if background_context:
                recipe = add_background_context(recipe, item_seed, negative_heavy=negative_heavy)
            if content_lines and recipe.get("background_context") not in ("blank","text_only"):
                recipe=add_content_lines(recipe,item_seed)
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
    parser.add_argument('--background-context', action='store_true', help='Mix page whitespace, text-only and blank negatives')
    parser.add_argument('--dense-text', action='store_true', help='Include 30%% paragraph-heavy forms')
    parser.add_argument('--dashed-borders', action='store_true', help='Vary internal dotted/dashed, thin and faint borders')
    parser.add_argument('--negative-heavy', action='store_true', help='With background context, increase text-only negatives to 30%%')
    parser.add_argument('--content-lines',action='store_true',help='Add text underlines and shallow diagonal negatives')
    parser.add_argument('--font', type=Path, action='append', help='Repeat for multiple Japanese fonts')
    args = parser.parse_args()
    fonts = args.font or [Path(__file__).resolve().parents[3]/'corpus/fonts'/name for name in
                         ('NotoSansCJKjp-Regular.otf', 'NotoSerifCJKjp-Regular.otf')]
    try:
        result = generate(args.output, args.count, args.seed, args.split, [f.resolve() for f in fonts], scale=args.scale, background_context=args.background_context, dense_text=args.dense_text, dashed_borders=args.dashed_borders, negative_heavy=args.negative_heavy, content_lines=args.content_lines)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
