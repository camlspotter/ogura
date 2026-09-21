from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf as fitz
from PIL import Image

from ogura.textdet.prepare import extract_page, filter_document, parse_fonts, prepare, inspect_fonts, trim_edge_space_chars, bounds
from ogura.textdet.filter_only import inspect_document, text_summary


FONT_OK = {'description': 'Fixture TrueType WinAnsi', 'to_unicode': True}


class PrepareTests(unittest.TestCase):
    def test_edge_spaces_trim_bbox_but_preserve_original_text(self):
        with fitz.open() as pdf:
            for rotation in (0, 90):
                page = pdf.new_page(width=300, height=300)
                page.insert_text((50, 100), '   A B  ', fontsize=15)
                page.set_rotation(rotation)
                line = extract_page(page)['lines'][0]
                self.assertEqual(line['text'], '   A B  ')
                self.assertEqual(''.join(c['text'] for c in line['chars']), '   A B  ')
                content = trim_edge_space_chars(line['chars'])
                self.assertEqual(''.join(c['text'] for c in content), 'A B')
                expected = bounds([p for c in content for p in c['polygon']])
                for actual, wanted in zip(line['bbox'], expected):
                    self.assertAlmostEqual(actual, wanted, places=4)

    def test_unicode_edge_spaces_and_internal_spaces(self):
        chars = [{'text': c} for c in '\u3000\tA \u3000B\u00a0 ']
        self.assertEqual(''.join(c['text'] for c in trim_edge_space_chars(chars)), 'A \u3000B')
        self.assertEqual(trim_edge_space_chars([{'text': '\u3000'}]), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.path = self.source / 'sample.pdf'
        with fitz.open() as pdf:
            page = pdf.new_page(width=300, height=400)
            page.insert_text((70, 100), 'Review sample', fontsize=16)
            page.set_cropbox(fitz.Rect(20, 30, 280, 380))
            page.set_rotation(90)
            pdf.new_page(width=200, height=200)
            pdf.save(self.path)

    def test_font_table_missing_mapping_and_invalid_format(self):
        fonts = parse_fonts('name type encoding emb sub uni object ID\n------- -------\n'
                            'FontA CID TrueType Identity-H yes yes yes 12 0\n'
                            'FontB Type 1 WinAnsi no no no 13 0\n')
        self.assertEqual([f['to_unicode'] for f in fonts], [True, False])
        self.assertEqual(filter_document(fonts, [], 100)[0]['code'], 'font_without_to_unicode')
        with self.assertRaises(ValueError):
            parse_fonts('not a table')

    def test_real_standard_font_without_to_unicode(self):
        fonts, _ = inspect_fonts(self.path)
        self.assertTrue(fonts)
        self.assertFalse(fonts[0]['to_unicode'])

    def test_filter_only_matches_character_check_without_rendering(self):
        with fitz.open(self.path) as pdf:
            for page in pdf:
                small, full = text_summary(page), extract_page(page)
                self.assertEqual(small['char_count'], full['char_count'])
                self.assertEqual(small['suspicious_characters'], full['suspicious_characters'])
        with patch('ogura.textdet.filter_only.inspect_fonts', return_value=([FONT_OK], '')), \
                patch.object(fitz.Page, 'get_pixmap', side_effect=AssertionError('Must not render')):
            record = inspect_document((self.path, 1))
        self.assertEqual(record['status'], 'passed')
        self.assertEqual(record['char_count'], 12)

    def test_filter_only_missing_font_does_not_extract_pages(self):
        with patch('ogura.textdet.filter_only.text_summary', side_effect=AssertionError('Unnecessary extraction')):
            result = inspect_document((self.path, 1))
        self.assertEqual(result['status'], 'skipped')
        self.assertFalse(result['text_checked'])

    def test_threshold_counts_whitespace_free_whole_document(self):
        pages = [{'page': 1, 'char_count': 60, 'suspicious_characters': []},
                 {'page': 2, 'char_count': 40, 'suspicious_characters': []}]
        self.assertEqual(filter_document([FONT_OK], pages, 100), [])
        self.assertEqual(filter_document([FONT_OK], pages, 101)[0]['code'], 'too_few_characters')
        pages[1]['suspicious_characters'] = [{'codepoint': 'U+FFFD', 'count': 1}]
        self.assertEqual(filter_document([FONT_OK], pages, 100)[0]['code'], 'unreadable_character_codes')

    def test_cropped_rotated_box_contains_rendered_ink(self):
        with fitz.open(self.path) as pdf:
            data = extract_page(pdf[0])
            pix = pdf[0].get_pixmap(colorspace=fitz.csGRAY, alpha=False)
        self.assertEqual((data['width'], data['height']), (350, 260))
        self.assertEqual(data['char_count'], 12)
        self.assertEqual(data['lines'][0]['text'], 'Review sample')
        x0, y0, x1, y1 = data['lines'][0]['bbox']
        ink = [(i % pix.width, i // pix.width) for i, value in enumerate(pix.samples) if value < 128]
        self.assertGreater(len(ink), 30)
        self.assertTrue(all(x0-1 <= x <= x1+1 and y0-1 <= y <= y1+1 for x, y in ink))

    def test_entire_pdf_skipped_with_one_missing_font(self):
        output = self.root / 'skipped'
        with patch('ogura.textdet.prepare.inspect_fonts', return_value=([FONT_OK, {'to_unicode': False}], '')):
            result = prepare(self.source, output, min_chars=1)
        self.assertEqual(result['counts']['skipped'], 1)
        self.assertFalse((output / 'sample').exists())
        self.assertEqual(list(output.glob('**/*.png')), [])

    def test_all_pages_rendered_including_blank_and_refuse_overwrite(self):
        output = self.root / 'passed'
        with patch('ogura.textdet.prepare.inspect_fonts', return_value=([FONT_OK], '')):
            result = prepare(self.source, output, min_chars=1, dpi=96)
        self.assertEqual(result['counts']['rendered_pages'], 2)
        for number in (1, 2):
            folder = output / 'sample' / f'page-{number:04d}'
            self.assertTrue((folder / 'labels.json').exists())
            with Image.open(folder / 'page.png') as a, Image.open(folder / 'bbox.png') as b:
                self.assertEqual(a.size, b.size)
        with self.assertRaises(FileExistsError):
            prepare(self.source, output)


if __name__ == '__main__':
    unittest.main()
