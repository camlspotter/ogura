import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ogura.tablerec.preview_table_cnn import comparison, error_image, select_samples
from ogura.tablerec.synth_table_cells import MODES


class SelectionTests(unittest.TestCase):
    def test_balances_modes_and_thin_lines_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for mode in MODES:
                for width in (0.75, 2):
                    for _ in range(3):
                        name = f'{len(records)}.json'
                        (root/name).write_text(json.dumps(dict(mode=mode, inner_line_width=width, outer_line_width=2)))
                        records.append(dict(recipe=name))
            class Dataset:
                def __len__(self):
                    return len(records)
            dataset = Dataset()
            dataset.root, dataset.records = root, records
            selected = select_samples(dataset, 20, 42)
            self.assertEqual(len({s['index'] for s in selected}), 20)
            self.assertEqual(selected, select_samples(dataset, 20, 42))
            for mode in MODES:
                for thin in (True, False):
                    self.assertEqual(sum(s['mode']==mode and s['thin']==thin for s in selected), 2)
            self.assertEqual(len(select_samples(dataset, 100, 42)), 30)
            with self.assertRaises(ValueError):
                select_samples(dataset, 0, 42)

    def test_error_polarity_and_original_resolution(self):
        target = np.array([[0.25,0.75]], dtype=np.float32)
        predicted = np.array([[0.75,0.25]], dtype=np.float32)
        error = error_image(predicted, target)
        self.assertEqual(error.getpixel((0,0)), (128,0,0))
        self.assertEqual(error.getpixel((1,0)), (0,0,128))
        im = Image.new('RGB',(20,30),'white')
        maps = np.zeros((2,30,20), dtype=np.float32)
        canvas = comparison(im, maps, maps)
        self.assertEqual(canvas.size, (60,162))
        self.assertEqual(canvas.getpixel((0,24)),(255,255,255))


if __name__ == '__main__':
    unittest.main()
