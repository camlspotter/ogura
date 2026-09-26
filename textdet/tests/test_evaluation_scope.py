import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from ogura.textdet.evaluation_scope import excluded_image, load_exclusions, prepare


class EvaluationScopeTests(unittest.TestCase):
    def test_document_match_is_exact_and_covers_every_page(self):
        policy = {'documents': ['public_document_ministry01376.pdf']}
        for n in (1, 6, 11, 16):
            self.assertTrue(excluded_image(f'public_document_ministry01376_page-{n:04d}_page.png', policy))
        self.assertFalse(excluded_image('public_document_ministry013760_page-0006_page.png', policy))
        with self.assertRaises(ValueError):
            excluded_image('../public_document_ministry01376_page-0006_page.png', policy)

    def test_subset_preserves_sources_and_checks_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; output = root/'subset'
            policy_path = root/'policy.json'
            policy_path.write_text(json.dumps(dict(schema_version=1, documents=['excluded.pdf'])))
            for split in ('val', 'test'):
                (source/split/'images').mkdir(parents=True)
                labels = {}
                for doc in ('kept', 'excluded'):
                    name = doc+'_page-0001_page.png'; content = (split+doc).encode()
                    (source/split/'images'/name).write_bytes(content)
                    labels[name] = dict(img_hash=hashlib.sha256(content).hexdigest())
                (source/split/'labels.json').write_text(json.dumps(labels))
            before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
            report = prepare(source, output, policy_path)
            self.assertEqual(report['counts'], dict(val=1, test=1))
            self.assertEqual(prepare(source, output, policy_path), report)
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertFalse((output/'val/images/excluded_page-0001_page.png').exists())
            (output/'val/images/kept_page-0001_page.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'image differs'):
                prepare(source, output, policy_path)
