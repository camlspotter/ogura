import unittest
from unittest.mock import patch
from pathlib import Path
from ogura.training.render import replace_unsupported, render_sample, Sample, RenderParams, BatchRenderer, Vocabulary
from tests.test_training import FONT


class SpaceNormalizationTests(unittest.TestCase):
    def test_missing_existing_and_edge_spaces(self):
        with patch('ogura.training.render.font_characters', return_value=frozenset(map(ord, ' 日本文\u3000'))):
            self.assertEqual(replace_unsupported('  日xy  本  ', 'font'), '日 本')
            self.assertEqual(replace_unsupported('xxx', 'font'), '')
            self.assertEqual(replace_unsupported('日\u3000\u3000本', 'font'), '日\u3000\u3000本')
            self.assertEqual(replace_unsupported('日  本', 'font'), '日 本')

    def test_mixed_font_edge_spaces(self):
        from ogura.training.render import normalized_text
        with patch('ogura.training.render.font_characters', return_value=frozenset(map(ord, ' A日本\u3000'))):
            p = RenderParams('jp', western_font_path='en')
            self.assertEqual(normalized_text('☃\u3000A☃日本☃ ', p), 'A 日本')

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_image_and_target_share_normalization(self):
        from PIL import ImageChops
        params = RenderParams(str(FONT))
        raw = '\U0010ffff日\U0010ffff\U0010fffe  本\U0010ffff'
        expected = '日 本'
        sample = Sample(raw, params)
        self.assertIsNone(ImageChops.difference(render_sample(sample),
                         render_sample(Sample(expected, params))).getbbox())
        batch = BatchRenderer(Vocabulary(' 日本'))([sample])
        self.assertEqual(batch.texts, [expected])
        self.assertEqual(batch.target_lengths.tolist(), [3])
        self.assertEqual(sample.text, raw)
