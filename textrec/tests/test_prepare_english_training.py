import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ogura.textrec.build_english import sha
from ogura.textrec.prepare_english_training import prepare


class MixTests(unittest.TestCase):
    def test_manifest_binding_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); jp=root/'jp'; en=root/'en'; jp.mkdir(); en.mkdir()
            (jp/'train.txt').write_text('日本語の文章です\n')
            (en/'train.txt').write_text('The office staff\n')
            (en/'validation.txt').write_text('Flowers are beautiful\n')
            validation=['短文です。','通常の検証文です。','これは長い行の検証文です。']
            chars=set('日本語の文章ですThe office staffFlowers are beautiful'+''.join(validation))
            (jp/'targets.jsonl').write_text(''.join(json.dumps(dict(character=c))+'\n' for c in sorted(chars)))
            excluded={str(jp/'train.txt'):sha(jp/'train.txt')}
            for name,text in zip(('validation','validation_short5','validation_long80'),validation):
                folder=root/'datasets'/name;folder.mkdir(parents=True)
                f=folder/'validation.txt';f.write_text(text+'\n');excluded[str(f)]=sha(f)
                (folder/'manifest.json').write_text(json.dumps(dict(training_text_sha256=sha(jp/'train.txt'),
                    targets_sha256=sha(jp/'targets.jsonl'),validation_text_sha256=sha(f),split='validation')))
            (en/'manifest.json').write_text(json.dumps(dict(targets_sha256=sha(jp/'targets.jsonl'),excluded_files=excluded,
                output_sha256={name:sha(en/name) for name in ('train.txt','validation.txt')})))
            with patch('ogura.textrec.prepare_english_training.ROOT',root):
                prepare(jp,en,root/'mixed')
                prepare(jp,en,root/'repeat')
                self.assertEqual((root/'mixed/train.txt').read_bytes(),(root/'repeat/train.txt').read_bytes())
                m=json.loads((root/'mixed/english/manifest.json').read_text())
                self.assertEqual(m['training_text_sha256'],sha(root/'mixed/train.txt'))
                self.assertEqual(m['samples'],1)
                with self.assertRaises(FileExistsError):prepare(jp,en,root/'mixed')
                (en/'train.txt').write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'English source changed'):prepare(jp,en,root/'bad')
