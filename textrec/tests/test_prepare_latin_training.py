import json
from pathlib import Path
import random
import tempfile
import unittest
from ogura.textrec.build_english import sha
from ogura.textrec.prepare_latin_training import prepare


class LatinMixTests(unittest.TestCase):
    def test_reproducible_mix_preserves_all_heldout_text_and_test_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';supp=root/'supp';suite=root/'suite'
            base.mkdir();supp.mkdir()
            aliases=root/'aliases.json';aliases.write_text('{"version":1,"groups":[]}')
            rng=random.Random(42)
            def text():return ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=24))
            (base/'train.txt').write_text(text()+'\n'+text()+'\n')
            targets=''.join(json.dumps({'character':c})+'\n' for c in 'abcdefghijklmnopqrstuvwxyz')
            (base/'targets.jsonl').write_text(targets)
            bm=dict(training_text_sha256=sha(base/'train.txt'),targets_sha256=sha(base/'targets.jsonl'))
            (base/'manifest.json').write_text(json.dumps(bm))
            excluded={str(base/'train.txt'):sha(base/'train.txt')};sources=[]
            for name, directory, split in [(n,base/n,'validation') for n in ('validation','english','validation_short5','validation_long80')]+[(('' if split=='validation' else 'test/')+n,suite/split/n,split) for split in ('validation','test') for n in ('quotes','homoglyphs')]:
                directory.mkdir(parents=True);value=text()
                (directory/'validation.txt').write_text(value+'\n')
                (directory/'validation.jsonl').write_text(json.dumps({'text':value})+'\n')
                (directory/'targets.jsonl').write_text(targets)
                m=dict(**bm,split=split,samples=1,min_length=24,max_length=24,validation_text_sha256=sha(directory/'validation.txt'))
                (directory/'manifest.json').write_text(json.dumps(m))
                excluded[str(directory/'validation.jsonl')]=sha(directory/'validation.jsonl')
                sources.append((name,directory,split))
            (supp/'train.txt').write_text(text()+'\n')
            sm=dict(targets_sha256=bm['targets_sha256'],aliases_sha256=sha(aliases),excluded_files=excluded,
                    output_sha256={'train.txt':sha(supp/'train.txt')})
            (supp/'manifest.json').write_text(json.dumps(sm))
            for name in ('one','two'):prepare(base,supp,suite,root/name,aliases)
            self.assertEqual((root/'one/train.txt').read_bytes(),(root/'two/train.txt').read_bytes())
            self.assertEqual(len((root/'one/train.txt').read_text().splitlines()),3)
            for name,source,split in sources:
                dest=root/'one'/name
                self.assertEqual((dest/'validation.txt').read_bytes(),(source/'validation.txt').read_bytes())
                self.assertEqual((dest/'validation.jsonl').read_bytes(),(source/'validation.jsonl').read_bytes())
                m=json.loads((dest/'manifest.json').read_text())
                self.assertEqual(m['split'],split)
                self.assertEqual(m['training_text_sha256'],sha(root/'one/train.txt'))
            with self.assertRaises(FileExistsError):prepare(base,supp,suite,root/'one',aliases)
            # A valid supplement hash cannot hide actual overlap with a held-out set.
            (supp/'train.txt').write_bytes((base/'english/validation.txt').read_bytes())
            sm['output_sha256']['train.txt']=sha(supp/'train.txt')
            (supp/'manifest.json').write_text(json.dumps(sm))
            with self.assertRaisesRegex(ValueError,'overlap'):prepare(base,supp,suite,root/'bad',aliases)
            self.assertFalse((root/'bad').exists())
