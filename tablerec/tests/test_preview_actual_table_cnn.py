import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from ogura.tablerec.preview_actual_table_cnn import report
from ogura.tablerec.table_cnn import TableUNet, CHANNELS


class ActualPreviewTests(unittest.TestCase):
    def test_invalid_scale(self):
        for scale in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'scale'):
                report(Path('missing'), Path('missing'), Path('missing'), scale=scale)
        with self.assertRaisesRegex(ValueError, 'together'):
            report(Path('missing'), Path('missing'), Path('missing'), scale=0.5, max_side=1000)

    def test_report_without_labels_and_with_transparency(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root/'images'
            images.mkdir()
            Image.new('RGBA', (64, 40), (0, 0, 0, 0)).save(images/'a&b.png')
            checkpoint = root/'model.pt'
            torch.save(dict(format_version=1, channels=list(CHANNELS),
                            base_channels=4, model=TableUNet(4).state_dict()), checkpoint)
            output = root/'preview'
            result = report(images, checkpoint, output, max_side=32)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(result['ground_truth'])
            self.assertNotIn('metrics', result['samples'][0])
            self.assertEqual(result['samples'][0]['inference_size'], (32, 20))
            with Image.open(output/'000-comparison.png') as image:
                self.assertEqual(image.size, (64, 44))
                self.assertEqual(image.getpixel((0, 24)), (255, 255, 255))
            self.assertIn('a&amp;b.png', (output/'index.html').read_text())
            self.assertEqual(json.loads((output/'summary.json').read_text())['count'], 1)
            scaled = report(images, checkpoint, root/'scaled', scale=0.5)
            self.assertEqual(scaled['samples'][0]['inference_size'], (32, 20))
            self.assertEqual(scaled['scale'], 0.5)
            with self.assertRaises(FileExistsError):
                report(images, checkpoint, output)


if __name__ == '__main__':
    unittest.main()
