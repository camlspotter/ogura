import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from ogura.textdet.diagnose_postprocess import sweep


class PostprocessTests(unittest.TestCase):
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
