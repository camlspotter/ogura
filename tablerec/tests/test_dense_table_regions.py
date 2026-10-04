import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ogura.tablerec.synth_table_cells import make_sample,add_dense_cells,render_sample,MODES
from ogura.tablerec.evaluate_table_regions import regions,measure

FONT=Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Regular.otf'


@unittest.skipUnless(FONT.exists(),'Japanese font required')
class DenseTests(unittest.TestCase):
    def test_layout_and_independent_error_regions(self):
        for i,mode in enumerate(MODES):
            original=make_sample(i,FONT,mode)
            before=copy.deepcopy(original)
            with patch('ogura.tablerec.synth_table_cells.Image.new',side_effect=AssertionError('JSON only')):
                recipe=add_dense_cells(original,i)
            self.assertEqual(before,original)
            self.assertTrue(any(len(c['lines'])>=3 for c in recipe['cells']))
            for cell in recipe['cells']:
                for run in cell['text_runs']:
                    self.assertGreaterEqual(run['xy'][1],cell['bbox'][1])
                    self.assertLess(run['xy'][1]+cell['font_size'],cell['bbox'][3])
            image,h,v=render_sample(json.loads(json.dumps(recipe)))
            sample=dict(recipe=recipe,image=image,horizontal=h,vertical=v)
            truth,masks=regions(sample)
            self.assertTrue(masks['text'].any())
            self.assertFalse((masks['text']&(truth.max(0)>0)).any())
            perfect=measure(truth,truth,masks)
            self.assertEqual(perfect['horizontal/text']['mean_probability'],0)
            self.assertEqual(perfect['horizontal/boundary']['recall_at_05'],1)
            broken=truth.copy();broken[:,masks['text']]=.8
            metrics=measure(broken,truth,masks)
            self.assertAlmostEqual(metrics['horizontal/text']['mean_probability'],.8)
            self.assertEqual(metrics['horizontal/text']['fraction_at_05'],1)
            self.assertEqual(metrics['horizontal/boundary']['recall_at_05'],1)
