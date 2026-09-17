import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ogura.recognize import binarize, load_image


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
