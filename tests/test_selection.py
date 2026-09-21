from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ogura.training.train import TrainConfig, train, sha256, selection_score, load_initial_weights
from ogura.training.model import make_model
from ogura.training.render import Vocabulary
from ogura.training.checkpoint import IdentityMismatch
from tests.test_training import FONT


class SelectionTests(unittest.TestCase):
    def test_equal_weight_not_character_weight(self):
        rows = [dict(kind='validation_length', mode='augmented', min_length=n, max_length=n, cer=c)
                for n,c in [(5,.03),(80,.006)]]
        self.assertAlmostEqual(selection_score('mean-augmented-cer', {'cer':.003}, rows), .013)
        with self.assertRaises(ValueError):
            selection_score('mean-augmented-cer', {'cer':.003}, rows[:1])

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_latest_warmstart_initial_best_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            text=root/'train.txt'; text.write_text('日本語\n文字\n')
            vocab=root/'targets.jsonl'
            vocab.write_text(''.join(json.dumps({'character':c})+'\n' for c in '日本語文字'))
            paths=[]
            for n in (5,25,80):
                d=root/str(n); d.mkdir()
                p=d/'validation.txt'; p.write_text(('日本語文字'*16)[:n]+'\n')
                (d/'targets.jsonl').write_bytes(vocab.read_bytes())
                (d/'manifest.json').write_text(json.dumps(dict(samples=1,min_length=n,max_length=n,
                    split='validation',training_text_sha256=sha256(text),targets_sha256=sha256(vocab),
                    validation_text_sha256=sha256(p))))
                paths.append(p)
            base=TrainConfig(text=text,vocabulary=vocab,font=FONT,run_dir=root/'old',
                channels=2,threads=1,batch_size=2,epochs=1,log_samples=0,device='cpu')
            with redirect_stdout(io.StringIO()):train(base)
            source=root/'old/latest.pt'
            initial=torch.load(source,weights_only=True)['model']
            cfg=replace(base,run_dir=root/'new',init_from=source,validation_text=paths[1],
                validation_augmented=True,monitor_validation=(paths[0],paths[2]),
                selection_metric='mean-augmented-cer',early_stopping_patience=2,epochs=4)
            # Initial CER .1; main improves but short/long regress, so mean rejects it.
            # Main evaluation runs twice (baseline and augmented).
            values=iter([.1,.1,.09,.09,.08,.08])
            def main_eval(*args):
                return dict(cer=next(values),exact_accuracy=.9,seconds=0)
            monitor_scores=iter([.1,.2,.3])
            def monitors(model,datasets,*args):
                c=next(monitor_scores)
                return [dict(info,cer=c,exact_accuracy=.9,seconds=0) for info,_ in datasets]
            with patch('ogura.training.train.evaluate',side_effect=main_eval), \
                 patch('ogura.evaluate_lengths.evaluate_sets',side_effect=monitors), redirect_stdout(io.StringIO()):
                self.assertEqual(train(cfg)['epoch'],2)
            best=torch.load(root/'new/best.pt',weights_only=True)
            self.assertEqual(best['epoch'],0)
            self.assertAlmostEqual(best['selection_score'],.1)
            self.assertEqual(len(best['validation_results']),6)
            for k,v in initial.items():self.assertTrue(torch.equal(v,best['model'][k]))
            before=(root/'new/metrics.jsonl').read_bytes()
            with redirect_stdout(io.StringIO()):
                train(replace(cfg,init_from=None,resume=True))
            self.assertEqual(before,(root/'new/metrics.jsonl').read_bytes())
            with self.assertRaises(IdentityMismatch):
                train(replace(cfg,init_from=None,resume=True,selection_metric='validation-cer'))
            model=make_model(6,2)
            with self.assertRaisesRegex(ValueError,'vocabulary'):
                load_initial_weights(model,source,Vocabulary('日本語文字'),2,vocabulary_path=text)


class MeanSetSelectionTests(SelectionTests):
    def test_latest_warmstart_initial_best_and_resume(self):
        original = train
        def with_new_metric(config, *args, **kwargs):
            if config.selection_metric == 'mean-augmented-cer':
                config = replace(config, selection_metric='mean-set-cer')
            return original(config, *args, **kwargs)
        with patch('tests.test_selection.train', side_effect=with_new_metric):
            super().test_latest_warmstart_initial_best_and_resume()
