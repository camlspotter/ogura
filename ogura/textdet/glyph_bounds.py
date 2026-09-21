"""Bounds of embedded vertical glyph outlines at their PDF draw positions."""
from collections import defaultdict

import pymupdf as fitz


def vertical_glyph_bounds(page):
    fonts = defaultdict(set)
    for xref, _, _, name, _, encoding, *_ in page.get_fonts(full=True):
        if encoding == 'Identity-V':
            fonts[name.split('+', 1)[-1]].add(xref)
    loaded = {}
    for name, xrefs in fonts.items():
        if len(xrefs) != 1:
            continue  # A trace font name alone cannot disambiguate font subsets.
        try:
            data = page.parent.extract_font(next(iter(xrefs)))[3]
            if data:
                loaded[name] = fitz.Font(fontbuffer=data)
        except (RuntimeError, ValueError):
            continue
    result = defaultdict(list)
    identity = fitz.mupdf.FzMatrix(1, 0, 0, 1, 0, 0)
    for span in page.get_texttrace():
        font = loaded.get(span['font'])
        if font is None or span['wmode'] != 1 or span['dir'] not in (
            (1., 0.), (0., -1.), (-1., 0.), (0., 1.)
        ):
            continue
        size = span['size']
        dx, dy = span['dir']
        for code, gid, origin, _ in span['chars']:
            if gid < 0 or not 0 <= code <= 0x10FFFF or chr(code).isspace():
                continue
            matrix = fitz.mupdf.FzMatrix(size * dx, size * dy, size * dy, -size * dx, *origin)
            try:
                path = fitz.mupdf.fz_outline_glyph(font.this, gid, matrix)
                # No stroke expansion. Keep the matrix alive across the low-level call.
                rect = fitz.mupdf.ll_fz_bound_path(path.m_internal, None, identity.internal())
                bbox = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1)
            except (RuntimeError, ValueError):
                continue
            if bbox.is_empty or bbox.is_infinite:
                continue
            # rawdict's vertical origin is displaced one em along the writing
            # direction relative to texttrace. Rotate this offset with the glyph.
            result[(span['font'], chr(code))].append(
                (origin[0] - size * dy, origin[1] + size * dx, size, bbox))
    return result


def lookup_bounds(index, span, char):
    matches = [bbox for x, y, size, bbox in index.get((span['font'], char['c']), [])
               if abs(x - char['origin'][0]) < .02 and abs(y - char['origin'][1]) < .02
               and abs(size - span['size']) < .02]
    if matches and all(bbox == matches[0] for bbox in matches):
        return matches[0]
    return None
