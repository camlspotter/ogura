import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from ogura.textdet.diagnose_postprocess import sweep, aggregate


class PostprocessTests(unittest.TestCase):
    def test_aggregate_uses_counts_and_keeps_models_and_settings_separate(self):
        base = dict(model='synth7000', bin_thresh=.3, unclip_ratio=0, box_thresh=.1)
        rows = [dict(base, matches=1, ground_truth=1, predictions=1),
                dict(base, matches=0, ground_truth=9, predictions=9),
                dict(base, model='synth9000', matches=8, ground_truth=10, predictions=10),
                dict(base, unclip_ratio=1.5, matches=2, ground_truth=10, predictions=10)]
        results = aggregate(rows)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['model'], 'synth9000')
        self.assertEqual(results[0]['f1'], .8)
        self.assertEqual(results[-1]['pages'], 2)
        self.assertEqual(results[-1]['f1'], .1)

    def test_threshold_separates_bridge_and_expansion_changes_boxes(self):
        probability = np.zeros((128,128), dtype=np.float32)
        probability[20:30,20:100] = .9
        probability[40:50,20:100] = .9
        probability[29:41,50:60] = .35
        gt = np.array([[20,20,100,30],[20,40,100,50]],dtype=np.float32)/128
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = sweep(probability, gt, Image.new('RGB',(128,128),'white'), root,
                         [.3,.5], [0,1.5], .1)
            self.assertEqual(len(rows),4)
            predictions = json.loads((root/'predictions.json').read_text())
            self.assertEqual(len(predictions['bin-0.3_unclip-0']),1)
            self.assertEqual(len(predictions['bin-0.5_unclip-0']),2)
            small = np.array(predictions['bin-0.5_unclip-0'])
            large = np.array(predictions['bin-0.5_unclip-1.5'])
            self.assertTrue(np.all((large[:,3]-large[:,1]) > (small[:,3]-small[:,1])))
            self.assertTrue((root/'opened-0.5.png').is_file())
