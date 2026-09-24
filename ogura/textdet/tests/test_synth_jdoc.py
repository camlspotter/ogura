import json
import os
from pathlib import Path
import tempfile
import unittest
from playwright.sync_api import sync_playwright
from ogura.textdet.synth_jdoc import ROOT, render_html


class TemplateTests(unittest.TestCase):
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
