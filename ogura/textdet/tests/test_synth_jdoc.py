import json
import os
from pathlib import Path
import tempfile
import unittest
from playwright.sync_api import sync_playwright
from collections import Counter
from ogura.textdet.synth_jdoc import ROOT, render_html, variation_plan, fixed_page_html, page_source
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
