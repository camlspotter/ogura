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

    def test_marks_change_input_but_never_masks_and_old_recipe_still_renders(self):
        recipe = make_sample(42, FONT, 'plain')
        recipe['degradation'] = {'kind': 'clean'}
        for cell in recipe['cells']:
            cell['marks'] = []
        recipe['line_styles'] = {'horizontal': {}, 'vertical': {}}
        recipe['boundary_appearance'] = {'horizontal': {}, 'vertical': {}}
        recipe['corner_radius'] = 0
        for item in recipe['cells']:
            item.pop('background_patch', None)
            item['text_stroke_width'] = 0
        baseline, h, v = render_sample(recipe)
        old = copy.deepcopy(recipe)
        old.update(schema_version=3, renderer='pillow-table-v3')
        for cell in old['cells']:
            del cell['marks']
            del cell['content_kind']
        for expected, actual in zip((baseline, h, v), render_sample(old)):
            self.assertEqual(expected.tobytes(), actual.tobytes())
        cell = recipe['cells'][0]
        x0, y0, x1, y1 = cell['bbox']
        variants = [dict(kind='diagonal', segments=segments, width=1, color='#111111')
                    for segments in ([[x0,y0,x1,y1]], [[x0,y1,x1,y0]],
                                     [[x0,y0,x1,y1],[x0,y1,x1,y0]])]
        variants += [dict(kind='checkbox', bbox=[x0+10,y0+10,x0+24,y0+24],
                          width=1, color='#111111', state=state)
                     for state in ('empty', 'checked', 'filled')]
        for mark in variants:
            cell['marks'] = [mark]
            actual, actual_h, actual_v = render_sample(recipe)
            self.assertNotEqual(baseline.tobytes(), actual.tobytes())
            self.assertEqual(h.tobytes(), actual_h.tobytes())
            self.assertEqual(v.tobytes(), actual_v.tobytes())

    def test_dashed_input_has_continuous_boundary_targets(self):
        recipe = make_sample(42, FONT, 'plain')
        recipe['degradation'] = {'kind': 'clean'}
        recipe['line_styles'] = {'horizontal': {}, 'vertical': {}}
        recipe['boundary_appearance'] = {'horizontal': {}, 'vertical': {}}
        recipe['corner_radius'] = 0
        for item in recipe['cells']:
            item.pop('background_patch', None)
            item['text_stroke_width'] = 0
        solid, h, v = render_sample(recipe)
        for channel, key in [('horizontal', 'y_boundaries'), ('vertical', 'x_boundaries')]:
            for coordinate in recipe[key][1:-1]:
                recipe['line_styles'][channel][str(coordinate)] = dict(dash=2, gap=4, phase=0.5)
        snapshot = copy.deepcopy(recipe)
        dashed, dh, dv = render_sample(json.loads(json.dumps(recipe)))
        self.assertEqual(recipe, snapshot)
        self.assertNotEqual(solid.tobytes(), dashed.tobytes())
        self.assertEqual(h.tobytes(), dh.tobytes())
        self.assertEqual(v.tobytes(), dv.tobytes())
        # A gap is visible in the input but remains supervised as a boundary.
        y = round(recipe['y_boundaries'][1])
        xs = recipe['x_boundaries']
        found = any(solid.getpixel((x,y)) != dashed.getpixel((x,y)) and dh.getpixel((x,y)) > 0
                    for y in range(y-2,y+3) for x in range(round(xs[0])+5, round(xs[1])-5))
        self.assertTrue(found)

    def test_rounding_arrows_background_changes_are_input_only(self):
        recipe = make_sample(42, FONT, 'plain')
        recipe['degradation'] = {'kind': 'clean'}
        recipe['corner_radius'] = 0
        before, h, v = render_sample(recipe)
        recipe['corner_radius'] = 5
        cell = recipe['cells'][0]
        x0, y0, x1, y1 = cell['bbox']
        cell['marks'] = [dict(kind='arrow', start=[x0+6,(y0+y1)/2],
                             end=[x1-6,(y0+y1)/2], head=3, double=True, width=1, color='#111111')]
        cell['background'] = '#ffcccc'
        cell['background_patch'] = dict(bbox=[x0,y0,(x0+x1)/2,y1], color='#ccffcc')
        snapshot = copy.deepcopy(recipe)
        after, h2, v2 = render_sample(json.loads(json.dumps(recipe)))
        self.assertEqual(snapshot, recipe)
        self.assertNotEqual(before.tobytes(), after.tobytes())
        self.assertEqual(h.tobytes(), h2.tobytes())
        self.assertEqual(v.tobytes(), v2.tobytes())
        self.assertTrue(all(value >= 250 for value in after.getpixel((round(x0),round(y0)))))
        for a, b in zip((after,h2,v2), render_sample(recipe)):
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_compact_geometry_and_content_coverage(self):
        kinds = set()
        compact = False
        for seed in range(30):
            recipe = make_sample(seed, FONT, 'mixed')
            kinds.update(c['content_kind'] for c in recipe['cells'])
            if recipe['geometry_profile'] == 'compact':
                compact = True
                self.assertTrue(any(14 <= b-a < 31 for a,b in zip(recipe['y_boundaries'],recipe['y_boundaries'][1:])))
                self.assertTrue(any(b-a >= 200 for a,b in zip(recipe['x_boundaries'],recipe['x_boundaries'][1:])))
                for cell in recipe['cells']:
                    for mark in cell['marks']:
                        if mark['kind'] == 'checkbox':
                            self.assertTrue(cell['bbox'][1] < mark['bbox'][1] < mark['bbox'][3] < cell['bbox'][3])
        self.assertTrue(compact)
        self.assertTrue({'empty_sign','arrow','checkbox','diagonal','hard_text','ordinary'} <= kinds)

    def test_templates_keep_content_in_appropriate_columns(self):
        for seed, expected in [(1,'checklist'), (5,'ledger')]:
            recipe = make_sample(seed, FONT, 'plain')
            self.assertEqual(recipe['template'], expected)
            for cell in recipe['cells']:
                row, column = cell['row'], cell['column']
                if row == 0:
                    continue
                if expected == 'checklist':
                    if column >= 2:
                        self.assertEqual(cell['content_kind'], 'checkbox')
                    if column == 1 or column == 3:
                        self.assertEqual(cell['background'], '#ffffff')
                    self.assertFalse(any(m['kind'] == 'arrow' for m in cell['marks']))
                elif column >= 1:
                    self.assertIn(cell['align'], ('right','center'))
                    self.assertFalse(cell['marks'])

    def test_ledger_arithmetic_and_checkbox_exclusivity(self):
        recipe = make_sample(5, FONT, 'plain')
        records = recipe['ledger_records']
        items = [r for r in records if r['label'] not in ('小計','合計')]
        for record in records:
            a,b,difference = record['values']
            self.assertEqual(difference,b-a)
        self.assertEqual(records[-1]['values'], [sum(r['values'][c] for r in items) for c in range(3)])
        recipe = make_sample(1,FONT,'plain')
        for row in range(1,recipe['rows']):
            answers = [m for cell in recipe['cells'] if cell['row']==row for m in cell['marks']]
            self.assertLessEqual(sum(m.get('state')=='checked' for m in answers),1)

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
                self.assertEqual(recipe['schema_version'], 8)
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
