from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from ogura.evaluate_lengths import font_grid_summary, validation_sets
from ogura.training.train import TrainConfig, train, sha256, identity_for
from ogura.training.render import Vocabulary
from ogura.training.checkpoint import IdentityMismatch
from tests.test_training import FONT

SECOND = FONT.with_name('NotoSerifCJKjp-Regular.otf')


class GridTests(unittest.TestCase):
    def test_macro_average_and_completeness(self):
        rows = [dict(mode='augmented', min_length=n, max_length=n, font=f,
                     font_sha256=f, cer=c, exact_accuracy=.9, seconds=1)
                for n,c in [(5,.06),(25,.03),(80,0.)] for f in ['a','b']]
        overall, summaries = font_grid_summary(rows, ['a','b'])
        self.assertAlmostEqual(overall['cer'], .03)
        self.assertEqual(overall['conditions'], 6)
        self.assertEqual(len(summaries),5)
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):font_grid_summary(bad,['a','b'])

    @unittest.skipUnless(FONT.exists() and SECOND.exists(), 'Two Noto fonts required')
    def test_grid_rendering_initial_best_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            text=root/'train.txt';text.write_text('日本語\n文字\n')
            vocab=root/'targets.jsonl'
            vocab.write_text(''.join(json.dumps(dict(character=c))+'\n' for c in ' 日本語文字'))
            paths=[]
            for n in (5,25,80):
                folder=root/str(n);folder.mkdir()
                p=folder/'validation.txt';p.write_text(('日本語文字'*16)[:n]+'\n')
                (folder/'targets.jsonl').write_bytes(vocab.read_bytes())
                (folder/'manifest.json').write_text(json.dumps(dict(samples=1,min_length=n,max_length=n,
                    split='validation',training_text_sha256=sha256(text),targets_sha256=sha256(vocab),
                    validation_text_sha256=sha256(p))))
                paths.append(p)
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,extra_fonts=(SECOND,),run_dir=root/'run',
                channels=2,threads=1,batch_size=2,epochs=3,log_samples=0,device='cpu',
                validation_text=paths[1],validation_augmented=True,monitor_validation=(paths[0],paths[2]),
                selection_metric='mean-font-cer',early_stopping_patience=1)
            identity=identity_for(cfg,torch.device('cpu'))
            datasets=validation_sets(cfg,Vocabulary.read(vocab),paths,identity,all_fonts=True)
            self.assertEqual(len(datasets),9) # six augmented plus three baselines
            for info,ds in datasets:
                self.assertEqual(ds[0],ds[0])
                if 'font' in info:
                    self.assertEqual(Path(ds[0].render_params.font_path).name,info['font'])
                    self.assertEqual(len(ds),1)
            from ogura.training.model import make_model
            source=root/'source.pt'
            torch.save(dict(characters=list(Vocabulary.read(vocab).characters),channels=2,
                            model=make_model(7,2).state_dict()),source)
            values=iter([.03,.02,.025])
            def evaluate_grid(model,sets,*args):
                score=next(values)
                return [dict(info,cer=score,exact_accuracy=.9,seconds=.01) for info,_ in sets]
            with patch('ogura.evaluate_lengths.evaluate_sets',side_effect=evaluate_grid),redirect_stdout(io.StringIO()):
                position=train(replace(cfg,init_from=source))
            self.assertEqual(position['epoch'],2)
            best=torch.load(root/'run/best.pt',weights_only=True)
            self.assertEqual(best['epoch'],1)
            self.assertAlmostEqual(best['selection_score'],.02)
            self.assertEqual(len([r for r in best['validation_results'] if r['kind']=='validation_font_mean']),2)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(train(replace(cfg,resume=True)),position)
            with self.assertRaises(IdentityMismatch):
                train(replace(cfg,resume=True,selection_metric='mean-augmented-cer'))
