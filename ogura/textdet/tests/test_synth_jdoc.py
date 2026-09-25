import json
import os
from pathlib import Path
import tempfile
import unittest
from playwright.sync_api import sync_playwright
from collections import Counter
from ogura.textdet.synth_jdoc import ROOT, render_html, variation_plan, fixed_page_html, page_source, extract_page, table_positions
from ogura.textdet.prepare_synth_texts import excerpt


class TemplateTests(unittest.TestCase):
    def test_balanced_reproducible_variations(self):
        args = (120, 42, ['sans','sans-bold','serif','serif-bold'], [12,16,20,24], [1.5,1.7,2], [0,.03,.08], .25)
        plan = variation_plan(*args)
        self.assertEqual(plan, variation_plan(*args))
        self.assertEqual(sum(p['vertical'] for p in plan), 30)
        self.assertEqual(set(Counter(p['font'] for p in plan).values()), {30})
        self.assertEqual(set(Counter(p['font_size'] for p in plan).values()), {30})
        self.assertEqual(set(Counter(p['columns'] for p in plan).values()), {40})
        self.assertEqual(len({(p['font'],p['font_size']) for p in plan}), 16)

    def test_excerpt_keeps_text_and_budget(self):
        text = '見出し\n\n' + '日本語の文章です。'*40 + '\n' + '続く段落です。'*30
        paragraphs = excerpt(text, 300)
        self.assertLessEqual(sum(map(len, paragraphs)), 300)
        self.assertGreater(sum(map(len, paragraphs)), 250)
        self.assertTrue(all(p in text for p in paragraphs))

    def test_shared_font_and_spacing(self):
        text, style = render_html({'title':'title','paragraphs':['text']}, Path('unused'),
                                 12, False, 2, 16, line_height=1.5, letter_spacing=.03,
                                 font_url='../fonts/font.otf')
        self.assertIn("url('../fonts/font.otf')", text)
        self.assertNotIn('base64', text)
        self.assertIn('letter-spacing:0.03em', text)
        self.assertEqual(style['line_height'],1.5)

    def test_escaped_content_and_offline_font(self):
        with tempfile.TemporaryDirectory() as tmp:
            font = Path(tmp)/'font.otf'
            font.write_bytes(b'font-test')
            record = {'title':'<title>', 'paragraphs':['<script>bad()</script> 日本語12文字']}
            first, style = render_html(record, font, 12, True, 2, 20)
            second, _ = render_html(record, font, 12, True, 2, 20)
            self.assertEqual(first, second)
            self.assertNotIn('<script>', first)
            self.assertNotIn('https://fonts.', first)
            self.assertIn('data:font/otf;base64,', first)
            self.assertIn('&lt;title&gt;', first)
            self.assertTrue(style['is_vertical'])


@unittest.skipUnless(os.getenv('RUN_SYNTH_BROWSER_TESTS') == '1', 'requires installed Chromium')
class BrowserLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def extract(self, markup):
        page = self.browser.new_page()
        page.set_content(markup)
        lines = page.evaluate((ROOT/'synth_lines.js').read_text())
        page.close()
        return lines

    def test_horizontal_wrap_and_edge_space(self):
        rows = self.extract('<p style="font:20px monospace;width:60px;word-break:break-all;white-space:pre-wrap">  ABCDEFGHIJKL  </p><p>   </p>')
        self.assertGreater(len(rows), 1)
        self.assertEqual(''.join(r['text'] for r in rows), 'ABCDEFGHIJKL')
        self.assertTrue(all(r['orientation']=='horizontal' and r['text']==r['text'].strip() for r in rows))

    def test_vertical_combined_digits_and_blocks(self):
        rows = self.extract('<style>p{writing-mode:vertical-rl;font:20px monospace;height:400px}.tcy{text-combine-upright:all}</style><p>縦書き<span class="tcy">12</span>文字</p><p>別段落</p>')
        self.assertEqual([r['text'] for r in rows], ['縦書き12文字', '別段落'])
        self.assertTrue(all(r['orientation']=='vertical' for r in rows))
        self.assertLess(rows[0]['bbox'][2]-rows[0]['bbox'][0], 40)

    def test_fixed_page_trims_dom_and_labels_in_both_directions(self):
        for vertical in (False, True):
            for columns in (1, 2, 3):
                for fraction in (1, .6):
                    with self.subTest(vertical=vertical, columns=columns, fraction=fraction):
                        record = {'title': 'Title', 'paragraphs': ['日本語の本文です。'*150]*6}
                        markup, _ = render_html(record, Path('unused'), 12, vertical, columns, 16,
                                                font_url='data:font/otf;base64,', line_height=1.7)
                        page = self.browser.new_page(viewport={'width':600,'height':800})
                        page.set_content(fixed_page_html(markup))
                        extract = (ROOT/'synth_lines.js').read_text()
                        before = page.evaluate(extract)
                        stats = page.evaluate((ROOT/'synth_paginate.js').read_text(),
                                              dict(lines=before, fraction=fraction))
                        after = page.evaluate(extract)
                        self.assertLess(len(after), len(before))
                        self.assertEqual(len(after)-1, stats['retained_lines'])
                        expected = ''.join(page.locator('h1, p').all_text_contents())
                        self.assertEqual(''.join(expected.split()), ''.join(''.join(l['text'] for l in after).split()))
                        x0,y0,x1,y1 = stats['content_bbox']
                        for line in after:
                            a,b,c,d = line['bbox']
                            self.assertTrue(0 <= a < c <= 600 and 0 <= b < d <= 800)
                            if line['element_id'] != 'title':
                                self.assertTrue(x0 <= a < c <= x1 and y0 <= b < d <= y1)
                        saved_html = page.content()
                        saved_image = page.screenshot()
                        page.set_content(fixed_page_html(markup))
                        optimized_lines, optimized_stats = extract_page(page, fraction)
                        self.assertEqual(optimized_lines, after)
                        self.assertEqual(optimized_stats, stats)
                        self.assertEqual(page.content(), saved_html)
                        self.assertEqual(page.screenshot(), saved_image)
                        page.close()


class PageSourceTests(unittest.TestCase):
    def test_combines_distinct_articles_and_retains_provenance(self):
        records = [dict(id=str(i), title=f'Title{i}', paragraphs=['本文'*30], source={'url':str(i)}) for i in range(5)]
        result = page_source(records, 4, 150)
        self.assertEqual([s['id'] for s in result['sources']], ['4','0','1'])
        self.assertEqual(result['paragraph_sources'], ['4','0','0','1','1'])
        self.assertIn('Title0', result['paragraphs'])
        with self.assertRaises(ValueError):
            page_source(records, 0, 10000)


@unittest.skipUnless(os.getenv('RUN_SYNTH_BROWSER_TESTS') == '1', 'requires installed Chromium')
class PaginationBoundaryTests(unittest.TestCase):
    setUpClass = classmethod(BrowserLabelTests.setUpClass.__func__)
    tearDownClass = classmethod(BrowserLabelTests.tearDownClass.__func__)
    def test_subpixel_boundary_does_not_drop_first_line(self):
        page = self.browser.new_page(viewport={'width':600,'height':800})
        record = {'title':'Title', 'paragraphs':['日本語の本文です。'*500]}
        markup, _ = render_html(record, Path('unused'), 12, False, 1, 16,
                                font_url='data:font/otf;base64,', line_height=1.5)
        page.set_content(fixed_page_html(markup))
        lines = page.evaluate((ROOT/'synth_lines.js').read_text())
        rect = page.locator('.content-body').bounding_box()
        first = next(l for l in lines if l['element_id'] != 'title')
        first['bbox'][1] = rect['y'] - .25
        stats = page.evaluate((ROOT/'synth_paginate.js').read_text(), dict(lines=lines, fraction=1))
        self.assertGreater(stats['retained_lines'], 0)
        first['bbox'][1] = rect['y'] - 2
        with self.assertRaisesRegex(Exception, 'No complete body line fits.*first'):
            page.evaluate((ROOT/'synth_paginate.js').read_text(), dict(lines=lines, fraction=1))
        page.close()


@unittest.skipUnless(os.getenv('RUN_SYNTH_BROWSER_TESTS') == '1', 'requires installed Chromium')
class TableLabelTests(unittest.TestCase):
    setUpClass = classmethod(BrowserLabelTests.setUpClass.__func__)
    tearDownClass = classmethod(BrowserLabelTests.tearDownClass.__func__)

    def test_cells_keep_independent_lines_and_survive_body_trimming(self):
        for position in ("top", "bottom"):
            for seed in range(20260924, 20260932):
                with self.subTest(seed=seed, position=position):
                    record = {'title':'集計結果', 'paragraphs':['日本語の本文です。'*400]*3}
                    markup, style = render_html(record, Path('unused'), seed, False, 2, 16,
                        font_url='data:font/otf;base64,', line_height=1.7, tables=True, table_position=position)
                    page = self.browser.new_page(viewport={'width':900,'height':1400})
                    page.set_content(fixed_page_html(markup))
                    original_table = page.locator('.table-block').inner_html()
                    lines, pagination = extract_page(page, .6)
                    self.assertEqual(page.locator('.table-block').inner_html(), original_table)
                    table_rect = page.locator('.table-block').bounding_box()
                    body_rect = page.locator('.content-body').bounding_box()
                    if position == 'top':
                        self.assertLessEqual(table_rect['y']+table_rect['height'], body_rect['y'])
                    else:
                        self.assertLessEqual(body_rect['y']+body_rect['height'], table_rect['y'])
                    self.assertEqual(style['table']['position'], position)
                    table_lines = [l for l in lines if l['element_id'].startswith('table-cell-')]
                    self.assertGreater(len(table_lines), style['table']['rows']*4)
                    counts = Counter(l['element_id'] for l in table_lines)
                    self.assertGreater(max(counts.values()), 1)  # multiline cells remain separate lines
                    cell_info = page.locator('td p, th p').evaluate_all('''ps => ps.map(p => {
                        const r=p.parentElement.getBoundingClientRect();
                        return {id:p.dataset.id,text:p.textContent,box:[r.left,r.top,r.right,r.bottom]};
                    })''')
                    cells = {c['id']:c for c in cell_info}
                    for line in table_lines:
                        a,b,c,d = cells[line['element_id']]['box']
                        x0,y0,x1,y1 = line['bbox']
                        self.assertTrue(a-.5 <= x0 < x1 <= c+.5 and b-.5 <= y0 < y1 <= d+.5)
                    self.assertTrue(any(not c['text'].strip() for c in cells.values()))
                    for cell in cells.values():
                        if not cell['text'].strip():
                            self.assertNotIn(cell['id'], counts)
                    expected = ''.join(page.locator('h1, p, figcaption').all_text_contents())
                    self.assertEqual(''.join(expected.split()), ''.join(''.join(l['text'] for l in lines).split()))
                    for line in lines:
                        x0,y0,x1,y1 = line['bbox']
                        self.assertTrue(0 <= x0 < x1 <= 900 and 0 <= y0 < y1 <= 1400)
                    page.close()


class TablePositionTests(unittest.TestCase):
    def test_balanced_and_reproducible_positions(self):
        plan = table_positions(16, 1234, 'both')
        self.assertEqual(Counter(plan), {'top':8, 'bottom':8})
        self.assertEqual(plan, table_positions(16, 1234, 'both'))
        self.assertEqual(table_positions(5, 1234, 'bottom'), ['bottom']*5)
