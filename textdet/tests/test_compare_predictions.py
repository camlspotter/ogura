import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ogura.textdet.compare_predictions import compare, panel


class FakeDetector(torch.nn.Module):
    def forward(self, image, **kwargs):
        return {'preds': [{'words': np.array([[.1, .2, .8, .4, .9]], dtype=np.float32)}]}


class ComparisonTests(unittest.TestCase):
    def test_comparison_outputs_and_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'val/images').mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(root/'val/images/sample.png')
            (root/'val/labels.json').write_text(json.dumps({'sample.png': {
                'polygons': [[[10, 20], [80, 20], [80, 40], [10, 40]]]}}))
            torch.save({}, root/'weights.pt')
            args = argparse.Namespace(data=root/'val', checkpoint=root/'weights.pt', output=root/'out',
                                      device='cpu', amp=False, input_size=128, limit=None)
            with patch('doctr.models.detection.db_resnet34', side_effect=lambda **kw: FakeDetector()) as factory:
                compare(args)
                self.assertEqual(factory.call_count, 2)
                self.assertTrue(factory.call_args_list[0].kwargs['pretrained'])
                self.assertFalse(factory.call_args_list[1].kwargs['pretrained'])
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['models']['finetuned']['f1'], 1)
            self.assertFalse(summary['partial_run'])
            with Image.open(root/'out/sample_comparison.png') as image:
                self.assertEqual(image.size, (384, 164))
            self.assertTrue((root/'out/pages.csv').is_file())
            self.assertTrue((root/'out/predictions.json').is_file())
            with self.assertRaises(FileExistsError):
                compare(args)

    def test_boxes_align_with_header_offset(self):
        view = panel(Image.new('RGB', (101, 101), 'white'), [[.1,.2,.8,.4]], [], 'GT')
        self.assertEqual(view.getpixel((10, 56)), (0, 160, 64))
        self.assertEqual(view.getpixel((50, 66)), (255, 255, 255))
