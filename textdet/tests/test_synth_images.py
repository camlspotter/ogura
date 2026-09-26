import base64
from collections import Counter
import io
from itertools import product
import os
from pathlib import Path
import unittest
from PIL import Image
from playwright.sync_api import sync_playwright
from ogura.textdet.synth_images import image_plan, add_image, size_float_image
from ogura.textdet.synth_jdoc import render_html, fixed_page_html, extract_page


class ImagePlanTests(unittest.TestCase):
    def test_coverage_balance_and_repeatability(self):
        names = [f'asset-{i}.png' for i in range(200)]
        plan = image_plan(names, 400, 42, 1600, 'both')
        self.assertEqual(plan, image_plan(names, 400, 42, 1600, 'both'))
        self.assertEqual(set(Counter(p['file'] for p in plan).values()), {2})
        self.assertEqual(Counter(p['position'] for p in plan), dict(top=200, bottom=200))
        self.assertEqual(Counter(p['layout'] for p in plan), dict(float=200, block=200))


@unittest.skipUnless(os.getenv('RUN_SYNTH_BROWSER_TESTS') == '1', 'requires Chromium')
class ImageLayoutTests(unittest.TestCase):
    def test_vertical_end_float_has_gap_for_bold_glyph_bounds(self):
        font = Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Bold.otf'
        if not font.is_file():
            self.skipTest('requires NotoSansCJKjp-Bold.otf')
        buf = io.BytesIO()
        Image.new('RGB', (32, 32), 'green').save(buf, format='PNG')
        src = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        markup, style = render_html(dict(title='確認', paragraphs=['日本語の本文です。' * 200] * 8),
            font, 20260930 + 216, True, 1, 24, line_height=1.5)
        markup = add_image(markup, style, dict(src=src, height=352, alignment='center',
            position='top', layout='float', float_side='inline-end'))
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport=dict(width=1200, height=1600))
                page.set_content(fixed_page_html(markup))
                page.evaluate('document.fonts.ready')
                size_float_image(page, 352)
                lines, _ = extract_page(page, 1)
                r = page.locator('.asset-figure img').bounding_box()
                above = [l for l in lines if l['element_id'].isdigit() and
                         min(l['bbox'][2], r['x']+r['width']) > max(l['bbox'][0], r['x'])]
                self.assertTrue(above)
                self.assertGreater(min(r['y'] - l['bbox'][3] for l in above), 1)
            finally:
                browser.close()

    def test_visible_image_and_text_labels_do_not_overlap(self):
        buf = io.BytesIO()
        Image.new('RGB', (32, 32), 'green').save(buf, format='PNG')
        src = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                for vertical, tables in ((False, False), (True, False), (False, True)):
                    for position, layout in product(('top', 'bottom'), ('block', 'float')):
                        with self.subTest(vertical=vertical, tables=tables, position=position, layout=layout):
                            record = dict(title='Title', paragraphs=['日本語の本文です。' * 200] * 8)
                            markup, style = render_html(record, Path('unused'), 42, vertical, 2, 16,
                                font_url='data:font/otf;base64,', tables=tables, table_position='bottom')
                            markup = add_image(markup, style, dict(src=src, height=128, alignment='right', position=position, layout=layout,
                                float_side='inline-start' if position == 'top' else 'inline-end'))
                            page = browser.new_page(viewport=dict(width=1000, height=1400))
                            try:
                                page.set_content(fixed_page_html(markup))
                                if layout == 'float':
                                    size_float_image(page, 128)
                                lines, pagination = extract_page(page, 1)
                                r = page.locator('.asset-figure img').bounding_box()
                                self.assertEqual(page.locator('img').count(), 1)
                                self.assertTrue(page.locator('img').evaluate('img => img.complete && img.naturalWidth > 0'))
                                self.assertTrue(0 <= r['y'] < r['y'] + r['height'] <= 1400)
                                self.assertGreater(pagination['retained_lines'], 0)
                                for line in lines:
                                    a,b,c,d = line['bbox']
                                    self.assertTrue(0 <= a < c <= 1000 and 0 <= b < d <= 1400)
                                    self.assertFalse(min(c,r['x']+r['width']) > max(a,r['x']) and
                                                     min(d,r['y']+r['height']) > max(b,r['y']))
                                self.assertTrue(all(l['element_id'] != 'asset' for l in lines))
                                if layout == 'float':
                                    axis = 0 if vertical else 1
                                    start = r['x'] if vertical else r['y']
                                    end = start + (r['width'] if vertical else r['height'])
                                    self.assertTrue(any(l['element_id'].isdigit() and
                                        min(l['bbox'][axis+2], end) > max(l['bbox'][axis], start)
                                        for l in lines), 'Text must wrap alongside the image')
                                    self.assertEqual(page.locator('.content-body .asset-figure').count(), 1)
                            finally:
                                page.close()
            finally:
                browser.close()
