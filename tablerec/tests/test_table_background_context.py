import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import ImageChops

from ogura.tablerec.synth_table_cells import add_background_context,make_sample,render_sample,generate,TableCellDataset
from ogura.tablerec.mix_table_cells import mix

FONT=Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Regular.otf'


@unittest.skipUnless(FONT.is_file(),'requires Japanese font')
class BackgroundTests(unittest.TestCase):
    def test_negative_targets_and_page_translation(self):
        original=make_sample(42,FONT,'mixed')
        original['degradation']={'kind':'clean'}
        expected=copy.deepcopy(original)
        cases={}
        for seed in range(100):
            recipe=add_background_context(original,seed)
            cases.setdefault(recipe['background_context'],recipe)
        self.assertEqual(set(cases),{'table','page_table','text_only','blank'})
        self.assertEqual(original,expected)
        for kind,recipe in cases.items():
            snapshot=copy.deepcopy(recipe)
            image,h,v=render_sample(json.loads(json.dumps(recipe)))
            self.assertEqual(recipe,snapshot)
            self.assertEqual(image.size,(recipe['width'],recipe['height']))
            if kind in ('blank','text_only'):
                self.assertFalse(np.array(h).any())
                self.assertFalse(np.array(v).any())
                self.assertEqual(bool(np.any(np.array(image)!=255)),kind=='text_only')
            if kind=='page_table':
                _,old_h,old_v=render_sample(original)
                dx,dy=recipe['page_offset']
                bbox=(dx,dy,dx+original['width'],dy+original['height'])
                for old,new in zip((old_h,old_v),(h,v)):
                    self.assertIsNone(ImageChops.difference(old,new.crop(bbox)).getbbox())
                    self.assertEqual(int(np.array(old).sum()),int(np.array(new).sum()))
                self.assertGreater(recipe['height'],original['height'])

    def test_json_generation_and_equal_mix_preserve_recipes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('ogura.tablerec.synth_table_cells.Image.new',side_effect=AssertionError('no images in generation')):
                generate(root/'old',5,42,'train',[FONT])
                generate(root/'new',5,43,'train',[FONT],background_context=True)
            before={p.name:p.read_bytes() for p in (root/'new/recipes').glob('*.json')}
            metadata=mix([root/'old',root/'new'],root/'mixed')
            self.assertEqual(metadata['count'],10)
            data=TableCellDataset(root/'mixed')
            self.assertEqual(len(data),10)
            for record in data.records[5:]:
                self.assertEqual((data.root/record['recipe']).read_bytes(),before[record['recipe'].split('source-1-')[1]])
            sample=data[6]
            self.assertEqual(sample['image'].size,sample['horizontal'].size)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                mix([root/'old',root/'old'],root/'duplicate')
            self.assertFalse((root/'duplicate').exists())
            with self.assertRaises(FileExistsError):
                mix([root/'old',root/'new'],root/'mixed')
            generate(root/'other',1,44,'train',[FONT])
            with self.assertRaisesRegex(ValueError,'counts'):
                mix([root/'old',root/'other'],root/'unequal')
