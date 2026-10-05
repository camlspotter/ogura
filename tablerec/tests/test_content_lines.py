import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import ImageChops
from ogura.tablerec.synth_table_cells import make_sample,add_dense_cells,add_content_lines,add_background_context,render_sample
from ogura.tablerec.table_regions import regions
from ogura.tablerec.evaluate_table_regions import measure
FONT=Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Regular.otf'
@unittest.skipUnless(FONT.exists(),'Japanese font')
class ContentLinesTests(unittest.TestCase):
    def test_input_only_marks_and_separate_metrics(self):
        original=add_dense_cells(make_sample(45,FONT,'plain'),45)
        before=copy.deepcopy(original)
        with patch('ogura.tablerec.synth_table_cells.Image.new',side_effect=AssertionError('JSON only')):
            recipe=add_content_lines(original,45)
        self.assertEqual(original,before)
        _,a,b=render_sample(original)
        image,h,v=render_sample(json.loads(json.dumps(recipe)))
        for old,new in [(a,h),(b,v)]: self.assertIsNone(ImageChops.difference(old,new).getbbox())
        truth,masks=regions(dict(recipe=recipe,image=image,horizontal=h,vertical=v))
        for role in ['underline','shallow_diagonal']:
            self.assertTrue(masks[role].any())
            self.assertFalse((masks[role]&(truth.max(0)>0)).any())
            prediction=truth.copy();prediction[0,masks[role]]=1
            self.assertEqual(measure(prediction,truth,masks)[f'horizontal/{role}']['fraction_at_05'],1)
        for cell in recipe['cells']:
            for mark in cell['marks']:
                if mark.get('role')=='shallow_diagonal':
                    self.assertLessEqual(abs(mark['angle_degrees']),20)
                    self.assertGreater(mark['angle_degrees'],0)
                    self.assertEqual(mark['segments'],[cell['bbox']])
                    self.assertEqual(cell['text_runs'],[])
                    self.assertEqual(cell['text'],'')
                if mark.get('role'):
                    for x0,y0,x1,y1 in mark['segments']:
                        box=cell['bbox'];self.assertTrue(box[0]<=x0<x1<=box[2]);self.assertTrue(box[1]<=min(y0,y1)<=max(y0,y1)<=box[3])
        negative=add_background_context(recipe,0);negative['background_context']='text_only'
        image,h,v=render_sample(negative)
        self.assertFalse(np.array(h).any());self.assertFalse(np.array(v).any())
        _,masks=regions(dict(recipe=negative,image=image,horizontal=h,vertical=v))
        self.assertTrue(masks['underline'].any())
