from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from ogura.textrec.training.train import TrainConfig, train
from ogura.textrec.training.evaluate import evaluate
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
                with patch('ogura.textrec.training.train.evaluate',side_effect=scored),redirect_stdout(io.StringIO()):
                    return train(config)
            # Equal CER counts as non-improvement. A new best resets the count.
            result=run(cfg,[.5,.5,.4,.4,.6])
            self.assertEqual(result['epoch'],5)
            full=torch.load(root/'full/latest.pt',weights_only=True)
            self.assertEqual(full['best']['epoch'],3)
            self.assertEqual(full['best']['validation_results'][0]['cer'], .4)
            self.assertEqual(full['best']['validation_results'][0]['epoch'], 3)
            events=[json.loads(s) for s in (root/'full/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['kind'],'early_stop')
            self.assertEqual(events[-1]['stale_epochs'],2)
            interrupted=replace(cfg,run_dir=root/'interrupted',early_stopping_patience=0,epochs=4)
            run(interrupted,[.5,.5,.4,.4])
            checkpoint=root/'interrupted/latest.pt'
            state=torch.load(checkpoint,weights_only=True)
            state['identity']['training_code_sha256']='6be33a35ce25e831be7544cc45f0e4f632eab9438c2be3185bc6f9c804af3041'
            state['identity']['settings'].pop('model_type', None)
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


class BestSummaryTests(unittest.TestCase):
    def test_saved_and_legacy_metrics_use_best_epoch_only(self):
        from ogura.textrec.training.metrics import best_validation_results, print_best_validation
        best = dict(epoch=11, step=100, metrics=dict(cer=.001, exact_accuracy=.98))
        rows = [dict(kind='validation', epoch=11, step=100, **best['metrics']),
                dict(kind='validation_baseline', epoch=11, step=100, cer=.0005, exact_accuracy=.99),
                dict(kind='validation_length', epoch=11, step=100, dataset='short5', mode='augmented',
                     cer=.002, exact_accuracy=.97)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows +
                            [dict(kind='validation',epoch=16,step=200,cer=.02,exact_accuracy=.90)]))
            self.assertEqual(best_validation_results(best,path),rows)
            output=io.StringIO()
            with redirect_stdout(output):print_best_validation(best,path)
            self.assertIn('epoch=11 step=100',output.getvalue())
            self.assertIn('dataset=short5 mode=augmented accuracy=97.00% CER=0.2000%',output.getvalue())
            self.assertNotIn('90.00%',output.getvalue())
            path.unlink()
            self.assertEqual(best_validation_results(dict(best,validation_results=rows),path),rows)
            fallback=best_validation_results(best,path)
            self.assertEqual(len(fallback),1)
            self.assertEqual(fallback[0]['cer'],.001)
