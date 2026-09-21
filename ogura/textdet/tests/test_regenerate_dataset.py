import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pymupdf as fitz
from PIL import Image
from ogura.textdet.regenerate_dataset import pack, generate, validate_recipe


class RegenerationTests(unittest.TestCase):
    def test_portable_recipe_and_source_hash_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'original';source.mkdir(); experiment=root/'experiment';experiment.mkdir()
            with fitz.open() as pdf:
                page=pdf.new_page(width=144,height=144)
                page.insert_text((20,40),'Sample text')
                pdf.new_page(width=144,height=144)  # Not selected; must not be exported.
                pdf.save(source/'sample.pdf')
            image='sample_page-0001_page.png'
            manifest={'seed':1,'experimental_quality_holdout':[],'splits':{}}
            for split in ('train','val','test'):
                (experiment/split).mkdir()
                labels={image:{'img_dimensions':[300,300],'polygons':[[[0,0],[1,0],[1,1],[0,1]]]}} if split=='train' else {}
                (experiment/split/'labels.json').write_text(json.dumps(labels))
                manifest['splits'][split]={'items':[{'source_pdf':'sample.pdf','page':1,'split_group':'sample.pdf','image':image}] if split=='train' else []}
            (experiment/'manifest.json').write_text(json.dumps(manifest))
            recipe_path=root/'recipe.json'; recipe=pack(experiment,source,recipe_path)
            self.assertNotIn(str(source),recipe_path.read_text())
            moved=root/'elsewhere';moved.mkdir();shutil.copyfile(source/'sample.pdf',moved/'sample.pdf');shutil.rmtree(source)
            out=root/'result'; report=generate(recipe_path,moved,out,workers=1)
            self.assertEqual(report['status'],'complete');self.assertFalse(report['partial_run'])
            self.assertEqual(len(list((out/'train/images').glob('*.png'))),1)
            with Image.open(out/'train/images'/image) as im:self.assertEqual(im.size,(300,300))
            self.assertTrue((out/'train/review/sample_page-0001_bbox.png').exists())
            self.assertEqual(len(json.loads((out/'train/labels.json').read_text())[image]['polygons']),1)
            with self.assertRaises(FileExistsError):generate(recipe_path,moved,out,workers=1)
            (moved/'sample.pdf').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'PDF contents differ'):generate(recipe_path,moved,root/'bad',workers=1)
            self.assertFalse((root/'bad').exists())
            recipe['pages'].append(dict(recipe['pages'][0],page=2,image='sample_page-0002_page.png',split='val'))
            with self.assertRaisesRegex(ValueError,'crosses splits'):validate_recipe(recipe)
