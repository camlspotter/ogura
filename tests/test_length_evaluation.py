from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from ogura.evaluate_lengths import main, validation_sets
from ogura.training.render import Vocabulary
from ogura.training.train import TrainConfig, train, sha256, identity_for
from tests.test_training import FONT


@unittest.skipUnless(FONT.exists(), 'Noto font required')
class LengthEvaluationTests(unittest.TestCase):
    def test_monitoring_preserves_training_and_cli_reports_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = root/'train.txt'; text.write_text('日本語\n文字\n')
            vocab = root/'targets.jsonl'
            vocab.write_text(''.join(json.dumps({'character':c})+'\n' for c in ' 日本語文字'))
            val = root/'val'; val.mkdir()
            (val/'validation.txt').write_text('日本\n語文\n')
            (val/'targets.jsonl').write_bytes(vocab.read_bytes())
            (val/'manifest.json').write_text(json.dumps(dict(samples=2,min_length=2,max_length=2,split='validation',
                training_text_sha256=sha256(text), targets_sha256=sha256(vocab), validation_text_sha256=sha256(val/'validation.txt'))))
            cfg = TrainConfig(text=text,vocabulary=vocab,font=FONT,run_dir=root/'plain',
                validation_text=val/'validation.txt',channels=2,threads=1,batch_size=2,epochs=1,log_samples=0,
                device='cpu',font_size_min=28,font_size_max=40,vertical_full_range=True,validation_augmented=True)
            with redirect_stdout(io.StringIO()):
                train(cfg)
                train(replace(cfg,run_dir=root/'monitored',monitor_validation=(val/'validation.txt',)))
            plain=torch.load(root/'plain/latest.pt',weights_only=True)
            monitored=torch.load(root/'monitored/latest.pt',weights_only=True)
            for key,value in plain['model'].items():self.assertTrue(torch.equal(value,monitored['model'][key]))
            self.assertEqual(plain['best']['metrics']['cer'],monitored['best']['metrics']['cer'])
            events=[json.loads(s) for s in (root/'monitored/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([e['mode'] for e in events if e['kind']=='validation_length'],['baseline','augmented'])
            # Monitoring can be added when resuming without changing training identity.
            with redirect_stdout(io.StringIO()):train(replace(cfg,resume=True,monitor_validation=(val/'validation.txt',)))
            checkpoint=root/'plain/best.pt'; before=sha256(checkpoint)
            args=['evaluate_lengths','--checkpoint',str(checkpoint),'--validation-text',str(val/'validation.txt'),
                  '--device','cpu','--output',str(root/'report.json')]
            with patch('sys.argv',args), redirect_stdout(io.StringIO()):main()
            results=json.loads((root/'report.json').read_text())['results']
            self.assertEqual([r['mode'] for r in results],['baseline','augmented'])
            self.assertTrue(all(r['samples']==2 for r in results))
            self.assertEqual(before,sha256(checkpoint))
            with patch('sys.argv',args), self.assertRaises(FileExistsError):main()
            (val/'validation.txt').write_text('文字\n')
            with self.assertRaisesRegex(ValueError,'manifest'):
                validation_sets(cfg,Vocabulary(' 日本語文字'),[val/'validation.txt'],identity_for(cfg,torch.device('cpu')))
