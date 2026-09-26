import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ogura.textrec.recognize import binarize, load_image, adjust_contrast


class PreprocessingTests(unittest.TestCase):
    def test_dark_strokes_survive_light_background(self):
        pixels = np.full((12, 20), 230, dtype=np.uint8)
        pixels[:, 5:7] = 70
        actual = np.asarray(binarize(Image.fromarray(pixels)))
        self.assertTrue((actual[:, 5:7] == 0).all())
        self.assertTrue((actual[:, :5] == 255).all())

    def test_uniform_white_and_dimensions(self):
        self.assertEqual(binarize(Image.new('L', (4, 4), 255)).getextrema(), (255, 255))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'image.png'
            Image.new('RGB', (31, 20), (220, 240, 250)).save(p)
            before = p.read_bytes()
            x, widths = load_image(p)
            y, other_widths = load_image(p, 'otsu')
            self.assertEqual(x.shape, (1, 1, 48, 80))
            self.assertTrue(torch.equal(widths, other_widths))
            self.assertEqual(x.shape, y.shape)
            self.assertTrue((x[:, :, :, widths.item():] == 1).all())
            self.assertEqual(p.read_bytes(), before)


    def test_contrast_retains_grayscale_and_handles_uniform_image(self):
        gradient = np.tile(np.arange(50, 201, dtype=np.uint8), (48, 1))
        image = Image.fromarray(gradient)
        result = np.asarray(adjust_contrast(image, cutoff=0))
        self.assertEqual(result.min(), 0)
        self.assertEqual(result.max(), 255)
        self.assertGreater(len(np.unique(result)), 100)
        self.assertTrue((np.diff(result.astype(int), axis=1) >= 0).all())
        self.assertEqual(adjust_contrast(Image.new('L', (8, 8), 220)).getextrema(), (220, 220))
        for value in (-1, 50, float('nan'), float('inf')):
            with self.assertRaises(ValueError): adjust_contrast(image, value)

    def test_cli_compares_and_saves_three_inputs(self):
        import io
        import json
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from ogura.textrec.recognize import main
        from ogura.textrec.training.model import make_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picture = root/'input.png'
            Image.fromarray(np.tile(np.arange(50, 201, dtype=np.uint8), (48, 1))).save(picture)
            checkpoint = root/'best.pt'
            model = make_model(2, 2)
            torch.save(dict(characters=['A'], channels=2, model=model.state_dict()), checkpoint)
            output = root/'out.jsonl'
            saved = root/'inputs'
            args = ['recognize', '--checkpoint', str(checkpoint), '--device', 'cpu',
                    '--compare-preprocessing', '--save-inputs', str(saved),
                    '--output', str(output), str(picture)]
            torch.set_num_threads(1)
            with patch('sys.argv', args), redirect_stdout(io.StringIO()): main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r['preprocessing'] for r in rows], ['none', 'otsu', 'contrast'])
            self.assertEqual(rows[2]['contrast_cutoff'], 1.0)
            for mode in ('none', 'otsu', 'contrast'):
                with Image.open(saved/f'001-input-{mode}.png') as im:
                    x, widths = load_image(picture, mode)
                    self.assertTrue(np.array_equal(np.asarray(im), (x[0, 0].numpy()*255).round().astype(np.uint8)))
                    self.assertEqual(widths.item(), 151)
