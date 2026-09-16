from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from ogura.training.train import TrainConfig, train
from ogura.training.evaluate import evaluate
from tests.test_training import FONT


@unittest.skipUnless(FONT.exists(), 'Noto font required')
class EarlyStoppingTests(unittest.TestCase):
    def test_plateau_reset_resume_and_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'train.txt').write_text('日本語\n文字\n')
            (root/'val.txt').write_text('日本\n語文\n')
            (root/'targets.jsonl').write_text(''.join(json.dumps({'character':c})+'\n' for c in '日本語文字'))
            cfg=TrainConfig(text=root/'train.txt',vocabulary=root/'targets.jsonl',font=FONT,
                validation_text=root/'val.txt',run_dir=root/'full',device='cpu',channels=2,
                threads=1,batch_size=2,epochs=10,log_samples=0,early_stopping_patience=2)
            def run(config, scores):
                values=iter(scores)
                def scored(*args):return dict(evaluate(*args),cer=next(values))
                with patch('ogura.training.train.evaluate',side_effect=scored),redirect_stdout(io.StringIO()):
                    return train(config)
            # Equal CER counts as non-improvement. A new best resets the count.
            result=run(cfg,[.5,.5,.4,.4,.6])
            self.assertEqual(result['epoch'],5)
            full=torch.load(root/'full/latest.pt',weights_only=True)
            self.assertEqual(full['best']['epoch'],3)
            events=[json.loads(s) for s in (root/'full/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['kind'],'early_stop')
            self.assertEqual(events[-1]['stale_epochs'],2)
            interrupted=replace(cfg,run_dir=root/'interrupted',early_stopping_patience=0,epochs=4)
            run(interrupted,[.5,.5,.4,.4])
            checkpoint=root/'interrupted/latest.pt'
            state=torch.load(checkpoint,weights_only=True)
            state['identity']['training_code_sha256']='6be33a35ce25e831be7544cc45f0e4f632eab9438c2be3185bc6f9c804af3041'
            torch.save(state,checkpoint)
            resumed=replace(interrupted,resume=True,epochs=10,early_stopping_patience=2)
            self.assertEqual(run(resumed,[.6]),result)
            state=torch.load(checkpoint,weights_only=True)
            for k,v in full['model'].items():self.assertTrue(torch.equal(v,state['model'][k]))
            before=(root/'interrupted/metrics.jsonl').read_bytes()
            self.assertEqual(run(resumed,[]),result)
            self.assertEqual(before,(root/'interrupted/metrics.jsonl').read_bytes())
            # Increasing patience allows further training; a new best resets the wait.
            self.assertEqual(run(replace(resumed,early_stopping_patience=3,epochs=6),[.3])['epoch'],6)
            self.assertEqual(torch.load(checkpoint,weights_only=True)['best']['epoch'],6)

    def test_requires_validation_and_nonnegative_patience(self):
        with self.assertRaisesRegex(ValueError,'validation-text'):
            train(TrainConfig(early_stopping_patience=5))
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            train(TrainConfig(early_stopping_patience=-1))
