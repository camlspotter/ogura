import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from ogura.textdet.combine_synthetic import combine


class CombineTests(unittest.TestCase):
    def test_image_sets_require_layout_metadata_and_distinct_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = self.source(root, 'text', False)
            tables = self.source(root, 'tables', True)
            image_text = self.source(root, 'image', False)
            image_tables = self.source(root, 'image-table', True)
            kwargs = dict(image_text=image_text, image_tables=image_tables,
                          image_text_count=1, image_table_count=1)
            with self.assertRaisesRegex(ValueError, 'reviewed image assets'):
                combine(text, tables, root/'invalid', 1, 1, **kwargs)
            for source in (image_text, image_tables):
                path = source/'manifest.json'
                manifest = json.loads(path.read_text())
                manifest['settings'] = dict(image_assets={'leaf.png':'hash'}, image_float_fraction=.5)
                manifest['pages'][0]['image_asset'] = dict(layout='block')
                path.write_text(json.dumps(manifest))
            report = combine(text, tables, root/'mixed', 1, 1, **kwargs)
            self.assertEqual(report['counts'], dict(text=1, table=1, image=1, **{'image-table':1}, total=4))
            labels = json.loads((root/'mixed/labels.json').read_text())
            self.assertIn('image-table-synth-000001.png', labels)
            path = image_text/'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['pages'][0]['image_asset']['layout'] = 'float'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'float layout count'):
                combine(text, tables, root/'wrong-float', 1, 1, **kwargs)

    def test_add_heading_sets_without_name_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = self.source(root, 'text', False)
            tables = self.source(root, 'tables', True)
            headings = self.source(root, 'headings', False)
            heading_tables = self.source(root, 'heading-tables', True)
            with self.assertRaisesRegex(ValueError, 'decoration settings'):
                combine(text, tables, root/'invalid', 1, 1, headings, heading_tables, 1)
            for source in (headings, heading_tables):
                path = source/'manifest.json'
                manifest = json.loads(path.read_text())
                manifest['settings'] = dict(section_headings=True, colored_text=True, title_style='mixed')
                path.write_text(json.dumps(manifest))
            report = combine(text, tables, root/'mixed', 1, 1, headings, heading_tables, 1)
            self.assertEqual(report['counts'], {'text':1, 'table':1, 'heading':1, 'heading-table':1, 'total':4})
            labels = json.loads((root/'mixed/labels.json').read_text())
            self.assertEqual(len(labels), 4)
            self.assertIn('heading-table-synth-000001.png', labels)

    def source(self, root, name, table):
        source = root/name
        (source/'images').mkdir(parents=True)
        image = source/'images/synth-000001.png'
        Image.new('RGB',(16,16),'white').save(image)
        label = dict(img_dimensions=[16,16], img_hash=hashlib.sha256(image.read_bytes()).hexdigest(),
                     polygons=[[[1,2],[10,2],[10,8],[1,8]]])
        (source/'labels.json').write_text(json.dumps({image.name:label}))
        page = dict(image=image.name)
        if table: page['table']={'position':'bottom'}
        (source/'manifest.json').write_text(json.dumps(dict(status='complete',pages=[page])))
        return source

    def test_name_collisions_and_independent_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);text=self.source(root,'text',False);tables=self.source(root,'tables',True)
            original={p:p.read_bytes() for src in (text,tables) for p in src.rglob('*') if p.is_file()}
            output=root/'mixed'
            report=combine(text,tables,output,1,1)
            self.assertEqual(report['counts'],dict(text=1,table=1,total=2))
            labels=json.loads((output/'labels.json').read_text())
            self.assertEqual(set(labels),{'text-synth-000001.png','table-synth-000001.png'})
            self.assertEqual(labels['text-synth-000001.png']['polygons'],[[[1,2],[10,2],[10,8],[1,8]]])
            (output/'images/text-synth-000001.png').write_bytes(b'modified copy')
            self.assertTrue(all(p.read_bytes()==data for p,data in original.items()))
            with self.assertRaises(FileExistsError):combine(text,tables,output,1,1)

    def test_reject_incomplete_wrong_kind_and_corrupt_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);text=self.source(root,'text',False);tables=self.source(root,'tables',True)
            with self.assertRaisesRegex(ValueError,'expected 2'):combine(text,tables,root/'wrong-count',2,1)
            with self.assertRaisesRegex(ValueError,'wrong page type'):combine(tables,text,root/'wrong-kind',1,1)
            (tables/'images/synth-000001.png').write_bytes(b'corrupt')
            output=root/'corrupt'
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):combine(text,tables,output,1,1)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'failed')
            self.assertFalse((output/'labels.json').exists())
