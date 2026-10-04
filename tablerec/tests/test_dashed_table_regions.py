import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ogura.tablerec.synth_table_cells import make_sample,add_dense_cells,add_dashed_borders,render_sample,generate
from ogura.tablerec.evaluate_table_regions import regions,boundary_regions,measure

FONT=Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Regular.otf'


@unittest.skipUnless(FONT.exists(),'Japanese font required')
class DashedTests(unittest.TestCase):
    def test_ink_gap_and_solid_are_independent(self):
        original=make_sample(45,FONT,'plain')
        before=copy.deepcopy(original)
        recipe=add_dashed_borders(add_dense_cells(original,45),45)
        self.assertEqual(original,before)
        recipe['degradation']={'kind':'clean'}
        # Ensure long, unambiguous gaps independent of random pattern selection.
        for style in recipe['line_styles']['horizontal'].values():
            style.update(dash=6,gap=5,phase=0)
        image,h,v=render_sample(json.loads(json.dumps(recipe)))
        truth,masks=regions(dict(recipe=recipe,image=image,horizontal=h,vertical=v))
        classes=boundary_regions(recipe,truth)
        for name in ('solid','dash_ink','dash_gap'):
            self.assertTrue(classes[name][0].any(),name)
        self.assertFalse((classes['solid']&classes['dash_gap']).any())
        self.assertFalse((classes['dash_ink']&classes['dash_gap']).any())
        perfect=measure(truth,truth,masks,classes)
        self.assertEqual(perfect['horizontal/dash_gap']['recall_at_05'],1)
        missing_gap=truth.copy();missing_gap[classes['dash_gap']]=0
        result=measure(missing_gap,truth,masks,classes)
        self.assertEqual(result['horizontal/dash_gap']['recall_at_05'],0)
        self.assertEqual(result['horizontal/dash_ink']['recall_at_05'],1)
        self.assertEqual(result['horizontal/solid']['recall_at_05'],1)
        recipe['background_context']='text_only'
        self.assertFalse(any(mask.any() for mask in boundary_regions(recipe,truth).values()))

    def test_generation_does_not_allocate_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('ogura.tablerec.synth_table_cells.Image.new',side_effect=AssertionError('JSON only')):
                generate(Path(tmp)/'data',5,20261011,'test',[FONT],
                         dense_text=True,background_context=True,dashed_borders=True)
            for path in (Path(tmp)/'data/recipes').glob('*.json'):
                recipe=json.loads(path.read_text())
                self.assertEqual(recipe['dash_profile'],'varied-dashes-v1')
                self.assertTrue(recipe['line_styles']['horizontal'])
                render_sample(recipe)
