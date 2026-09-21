import unittest
from unittest.mock import Mock, patch

import pymupdf as fitz

from ogura.textdet.prepare import bounds, extract_page, merge_vertical_fragments, merge_rotated_vertical_prefixes
from ogura.textdet.glyph_bounds import lookup_bounds, vertical_glyph_bounds


def fragment(number, x, y, wmode=1, size=10):
    return {'id': number, 'source_line_ids': [number], 'text': str(number),
            'wmode': wmode, 'baseline': (0.0, 1.0), 'fonts': ['Fixture'],
            'chars': [{'text': str(number), 'size': size, 'origin': (x, y)}],
            'bbox': [x, y, x + size, y + size],
            'polygon': [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]}


class VerticalMergeTests(unittest.TestCase):
    def test_rotated_prefix_merges_by_position_across_blocks(self):
        for rotation in (0, 90):
            matrix = fitz.Matrix(rotation)
            prefix = fragment(1, 14, 10)
            prefix.update(text='－', baseline=(1., 0.), polygon=[[14, 10], [15, 10], [15, 18], [14, 18]])
            prefix['chars'][0].update(geometry_source='glyph_outline', text='－')
            following = fragment(2, 10, 20)
            following['chars'][0]['polygon'] = following['polygon']
            for line in (prefix, following):
                line['polygon'] = [[q.x, q.y] for q in (fitz.Point(p) * matrix for p in line['polygon'])]
            following['chars'][0]['polygon'] = following['polygon']
            result = merge_rotated_vertical_prefixes([prefix, following], ~matrix)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['text'], '－2')
            self.assertEqual(result[0]['source_line_ids'], [1, 2])
            self.assertEqual(result[0]['baseline'], (0., 1.))

    def test_rotated_prefix_rejects_other_column_gap_and_ambiguity(self):
        for positions in ([(30, 20)], [(10, 40)], [(10, 20), (11, 20)]):
            prefix = fragment(1, 14, 10)
            prefix.update(baseline=(1., 0.), polygon=[[14, 10], [15, 10], [15, 18], [14, 18]])
            prefix['chars'][0]['geometry_source'] = 'glyph_outline'
            lines = [prefix]
            for x, y in positions:
                line = fragment(len(lines) + 1, x, y)
                line['chars'][0]['polygon'] = line['polygon']
                lines.append(line)
            self.assertEqual(len(merge_rotated_vertical_prefixes(lines, fitz.Matrix(1, 1))), len(lines))

    def test_embedded_glyph_bounds_cover_rendered_ink(self):
        font = fitz.Font('helv')
        for rotation, direction, raw_origin in ((0, (1., 0.), (40., 70.)),
                                                 (90, (0., -1.), (60., 50.)),
                                                 (180, (-1., 0.), (40., 30.)),
                                                 (270, (0., 1.), (20., 50.))):
            trace = {'font': 'Fixture', 'wmode': 1, 'dir': direction, 'size': 20,
                     'chars': [(ord('A'), font.has_glyph(ord('A')), (40., 50.), ())]}
            source = Mock()
            source.get_fonts.return_value = [(1, 'ttf', 'Type0', 'Fixture', 'F1', 'Identity-V')]
            source.parent.extract_font.return_value = ('Fixture', 'ttf', 'Type0', font.buffer)
            source.get_texttrace.return_value = [trace]
            index = vertical_glyph_bounds(source)
            rect = lookup_bounds(index, {'font': 'Fixture', 'size': 20},
                                 {'c': 'A', 'origin': raw_origin})
            self.assertIsNotNone(rect)
            with fitz.open() as pdf:
                page = pdf.new_page(width=100, height=100)
                page.insert_font(fontname='Fixture', fontbuffer=font.buffer)
                page.insert_text((40, 50), 'A', fontname='Fixture', fontsize=20, rotate=rotation)
                pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), colorspace=fitz.csGRAY)
                pixels = [(i % pix.width / 4, i // pix.width / 4)
                          for i, value in enumerate(pix.samples) if value < 128]
                ink = bounds(pixels)
                for actual, expected in zip(rect, ink):
                    self.assertLess(abs(actual - expected), 1.)

    def test_vertical_outline_bounds_replace_metrics_and_rotate(self):
        bbox = (30, 40, 41.3, 51.3)
        span = {'font': 'Fixture', 'size': 11.3, 'ascender': 1.194,
                'descender': -0.325, 'bbox': bbox, 'origin': (30, 51.3),
                'chars': [{'c': 'み', 'bbox': bbox, 'origin': (30, 51.3)}]}
        raw = {'blocks': [{'type': 0, 'lines': [{'wmode': 1, 'dir': (0., 1.),
                                               'bbox': bbox, 'spans': [span]}]}]}
        with fitz.open() as pdf:
            for rotation in (0, 90):
                page = pdf.new_page(width=100, height=200)
                page.set_rotation(rotation)
                proxy = Mock(wraps=page)
                proxy.get_text.return_value = raw
                proxy.rect = page.rect
                proxy.rotation_matrix = page.rotation_matrix
                proxy.rotation = rotation
                proxy.number = page.number
                with patch('ogura.textdet.prepare.vertical_glyph_bounds', return_value={}), patch('ogura.textdet.prepare.lookup_bounds', return_value=fitz.Rect(bbox)):
                    line = extract_page(proxy)['lines'][0]
                expected = list(fitz.Rect(bbox) * page.rotation_matrix)
                for actual in (line['bbox'], bounds(line['chars'][0]['polygon'])):
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b, places=4)

    def test_column_fragments_merge_and_preserve_characters(self):
        lines = merge_vertical_fragments([fragment(1, 10, 10), fragment(2, 10, 20), fragment(3, 10, 30)])
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]['text'], '123')
        self.assertEqual(lines[0]['source_line_ids'], [1, 2, 3])
        self.assertEqual(lines[0]['bbox'], [10, 10, 20, 40])
        self.assertEqual(len(lines[0]['chars']), 3)

    def test_other_column_gap_size_or_horizontal_remains_separate(self):
        for following in (fragment(2, 20, 20), fragment(2, 10, 40),
                          fragment(2, 10, 20, size=20), fragment(2, 10, 20, wmode=0)):
            with self.subTest(following=following):
                self.assertEqual(len(merge_vertical_fragments([fragment(1, 10, 10), following])), 2)
