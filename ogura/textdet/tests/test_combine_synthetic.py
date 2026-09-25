import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from ogura.textdet.combine_synthetic import combine


class CombineTests(unittest.TestCase):
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
