import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import copy
import shutil

from PIL import Image, ImageChops

from ogura.tablerec.synth_table_cells import (
    MODES, TableCellDataset, boundary_segments, draw_mask, generate, make_sample, partition, render_sample, sample_seed,
)

FONT = Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Regular.otf'


class StructureTests(unittest.TestCase):
    def test_partitions_cover_grid_once_and_expose_all_boundaries(self):
        for mode in MODES:
            for seed in range(100):
                cells = partition(8, 6, random.Random(seed), mode)
                covered = [(y, x) for c in cells
                           for y in range(c['row'], c['row']+c['rowspan'])
                           for x in range(c['column'], c['column']+c['colspan'])]
                self.assertEqual(len(covered), 48)
                self.assertEqual(set(covered), {(y, x) for y in range(8) for x in range(6)})
                self.assertEqual({v for c in cells for v in (c['row'], c['row']+c['rowspan'])}, set(range(9)))
                self.assertEqual({v for c in cells for v in (c['column'], c['column']+c['colspan'])}, set(range(7)))
                if mode == 'plain':
                    self.assertEqual(len(cells), 48)
                elif mode == 'block_merge':
                    self.assertTrue(any(c['rowspan'] > 1 and c['colspan'] > 1 for c in cells))

    def test_merged_interior_is_not_drawn_and_crossing_has_both_channels(self):
        cells = [dict(row=0, column=0, rowspan=1, colspan=2),
                 dict(row=1, column=0, rowspan=1, colspan=1),
                 dict(row=1, column=1, rowspan=1, colspan=1)]
        hs, vs = boundary_segments(cells, [10, 40, 70], [10, 40, 70])
        h = draw_mask((80, 80), hs, 1)
        v = draw_mask((80, 80), vs, 1)
        self.assertEqual(v.getpixel((80, 50)), 0)
        self.assertEqual(v.getpixel((80, 110)), 255)
        self.assertEqual(h.getpixel((80, 80)), 255)
        self.assertEqual(v.getpixel((80, 80)), 255)
        self.assertEqual(h.getpixel((50, 50)), 0)

    def test_fractional_stroke_coverage_and_translation(self):
        a = draw_mask((12, 12), [(2, 5.25, 10, 5.25)], 0.75, scale=4, subpixel=True)
        b = draw_mask((12, 12), [(2, 5.5, 10, 5.5)], 0.75, scale=4, subpixel=True)
        a, b = [im.resize((12, 12), Image.Resampling.BOX) for im in (a, b)]
        self.assertIsNotNone(ImageChops.difference(a, b).getbbox())
        for mask in (a, b):
            self.assertAlmostEqual(sum(mask.getpixel((6, y)) for y in range(12))/255, 0.75, delta=0.01)
            self.assertTrue(any(0 < value < 255 for value in mask.tobytes()))
        self.assertEqual(a.getpixel((1, 5)), 0)
        self.assertEqual(a.getpixel((10, 5)), 0)

    def test_split_seeds_are_repeatable_and_disjoint(self):
        a = [sample_seed(42, 'train', i) for i in range(100)]
        b = [sample_seed(42, 'validation', i) for i in range(100)]
        self.assertFalse(set(a) & set(b))
        self.assertEqual(a, [sample_seed(42, 'train', i) for i in range(100)])


@unittest.skipUnless(FONT.is_file(), 'requires local Japanese font')
class RenderTests(unittest.TestCase):
    def test_render_repeatable_and_masks_match_geometry(self):
        for mode in MODES:
            labels = make_sample(1234, FONT, mode)
            labels2 = json.loads(json.dumps(labels))
            with patch('ogura.tablerec.synth_table_cells.random.Random', side_effect=AssertionError('renderer must not use RNG')):
                image, h, v = render_sample(labels)
                image2, h2, v2 = render_sample(labels2)
            self.assertEqual(json.loads(json.dumps(labels)), labels2)
            for a, b in zip((image, h, v), (image2, h2, v2)):
                self.assertIsNone(ImageChops.difference(a, b).getbbox())
                self.assertEqual(a.size, (labels['width'], labels['height']))
            xs, ys = labels['x_boundaries'], labels['y_boundaries']
            for cell in labels['cells']:
                r, c, rs, cs = (cell[k] for k in ('row', 'column', 'rowspan', 'colspan'))
                self.assertEqual(cell['bbox'], [xs[c], ys[r], xs[c+cs], ys[r+rs]])
                for x in range(c+1, c+cs):
                    self.assertEqual(v.getpixel((int(xs[x]), int((ys[r]+ys[r+1])//2))), 0)
                for y in range(r+1, r+rs):
                    self.assertEqual(h.getpixel((int((xs[c]+xs[c+1])//2), int(ys[y]))), 0)
                self.assertGreater(h.getpixel((int((xs[c]+xs[c+1])//2), int(ys[r]))), 0)
                self.assertGreater(v.getpixel((int(xs[c]), int((ys[r]+ys[r+1])//2))), 0)

    def test_dataset_contract_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'data'
            with patch('ogura.tablerec.synth_table_cells.Image.new', side_effect=AssertionError('JSON generation must not create images')):
                result = generate(output, 5, 42, 'train', [FONT])
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['modes'], dict.fromkeys(MODES, 1))
            records = [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual({p.suffix for p in output.rglob('*') if p.is_file()}, {'.json', '.jsonl'})
            for record in records:
                recipe = json.loads((output/record['recipe']).read_text())
                self.assertEqual(recipe['seed'], record['seed'])
                self.assertEqual(recipe['schema_version'], 3)
            dataset = TableCellDataset(output)
            self.assertEqual(len(dataset), 5)
            before = {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            sample = dataset[1]
            self.assertEqual(sample['horizontal'].mode, 'L')
            self.assertEqual(sample['image'].size, sample['vertical'].size)
            self.assertEqual(before, {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()})
            original = (output/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                generate(output, 5, 42, 'train', [FONT])
            self.assertEqual(original, (output/'manifest.json').read_bytes())
            with self.assertRaises(ValueError):
                generate(Path(tmp)/'invalid', 0, 42, 'train', [FONT])
            self.assertFalse((Path(tmp)/'invalid').exists())

    def test_recipe_edits_control_rendering_and_input_is_not_mutated(self):
        recipe = make_sample(42, FONT, 'plain')
        recipe['degradation'] = {'kind': 'clean'}
        snapshot = copy.deepcopy(recipe)
        original, h, v = render_sample(recipe)
        self.assertEqual(recipe, snapshot)
        recipe['line_color'] = '#ff0000'
        edited, h2, v2 = render_sample(recipe)
        self.assertIsNotNone(ImageChops.difference(original, edited).getbbox())
        self.assertIsNone(ImageChops.difference(h, h2).getbbox())
        self.assertIsNone(ImageChops.difference(v, v2).getbbox())
        for cell in recipe['cells']:
            cell['text_runs'] = []
        no_text, _, _ = render_sample(recipe)
        self.assertIsNotNone(ImageChops.difference(edited, no_text).getbbox())

    def test_scales_and_fractional_parameters(self):
        for scale in (2, 4, 8):
            recipe = make_sample(42, FONT, 'plain', scale=scale)
            self.assertEqual(recipe['scale'], scale)
            self.assertTrue(any(x != int(x) for x in recipe['x_boundaries']))
            image, h, v = render_sample(json.loads(json.dumps(recipe)))
            self.assertEqual(image.size, (recipe['width'], recipe['height']))
            self.assertEqual(h.size, v.size)
            self.assertTrue(any(0 < value < 255 for value in h.tobytes()))
        with self.assertRaises(ValueError):
            make_sample(42, FONT, 'plain', scale=3)
        recipe['scale'] = 3
        with self.assertRaises(ValueError):
            render_sample(recipe)

    def test_portable_font_resolution_and_wrong_font_rejection(self):
        recipe = make_sample(42, FONT, 'plain')
        expected, _, _ = render_sample(recipe)
        recipe['font']['path'] = '/nonexistent/original/font.otf'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copyfile(FONT, root/FONT.name)
            actual, _, _ = render_sample(recipe, font_dir=root)
            self.assertIsNone(ImageChops.difference(expected, actual).getbbox())
            recipe['font']['sha256'] = 'incorrect'
            with self.assertRaisesRegex(ValueError, 'Font content differs'):
                render_sample(recipe, font_dir=root)
        recipe['schema_version'] = 999
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            render_sample(recipe)


if __name__ == '__main__':
    unittest.main()
