from dataclasses import replace
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import random
import tempfile
import unittest
import warnings

import torch

from ogura.build_validation import extract
from ogura.diversity import fingerprints
from ogura.text_common import split_article
from ogura.training.checkpoint import rng_state
from ogura.training.evaluate import evaluate
from ogura.training.model import LineCNN
from ogura.training.render import Vocabulary
from ogura.training.train import TrainConfig, EpochDataset, train
from tests import test_training as helpers

FONT = helpers.FONT


class ExtractionTests(unittest.TestCase):
    def test_split_provenance_training_exclusion_and_repeatability(self):
        rng=random.Random(17)
        alphabet='東京大阪北海道日本世界歴史文化美術科学研究'
        articles=[]
        for i in range(400):
            text=''.join(rng.choices(alphabet,k=8))+'について'+''.join(rng.choices(alphabet,k=8))+'説明しています。'
            articles.append(dict(id=str(i),text=text,title='本文',url='example'))
        eligible=[a for a in articles if split_article(a['id'],20260915)=='validation']
        forbidden=fingerprints(eligible[0]['text'][:25])
        vocab=set(''.join(a['text'] for a in articles))
        first,_=extract(iter(articles),vocab,forbidden,5,7)
        second,_=extract(iter(articles),vocab,forbidden,5,7)
        self.assertEqual(first,second)
        self.assertEqual(len({r['article_id'] for r in first}),5)
        for row in first:
            self.assertEqual(split_article(row['article_id'],20260915),'validation')
            self.assertTrue(forbidden.isdisjoint(fingerprints(row['text'])))
            original=articles[int(row['article_id'])]['text']
            self.assertEqual(original[row['start']:row['end']],row['text'])
        with self.assertRaises(ValueError):extract([],vocab,set(),1,7)


@unittest.skipUnless(FONT.exists(), 'Noto font required')
class EvaluationTests(unittest.TestCase):
    def test_evaluation_is_fixed_and_does_not_change_model_or_rng(self):
        torch.set_num_threads(1)
        config=TrainConfig(font=FONT)
        vocab=Vocabulary('日本語文字')
        data=EpochDataset([('日本語','a'),('文字','b')],[0,1],config,epoch=0)
        model=LineCNN(len(vocab),channels=2)
        model.train()
        before={k:v.clone() for k,v in model.state_dict().items()}
        rng_before=rng_state()
        a=evaluate(model,data,vocab,2,torch.device('cpu'))
        b=evaluate(model,data,vocab,2,torch.device('cpu'))
        self.assertTrue(model.training)
        helpers.TrainingTests.assert_nested_equal(self,before,model.state_dict())
        helpers.TrainingTests.assert_nested_equal(self,rng_before,rng_state())
        for key in a:
            if key not in ('seconds','samples_per_second'):self.assertEqual(a[key],b[key])

    # Reuse the recursive comparator, without inheriting the other integration tests.
    assert_nested_equal = helpers.TrainingTests.assert_nested_equal

    def test_best_model_and_validation_history_survive_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'train.txt').write_text('日本語\n文字\n')
            (root/'val.txt').write_text('日本\n語文字\n')
            (root/'targets.jsonl').write_text(''.join(json.dumps({'character':c},ensure_ascii=False)+'\n' for c in '日本語文字'))
            config=TrainConfig(text=root/'train.txt',validation_text=root/'val.txt',
                               vocabulary=root/'targets.jsonl',font=FONT,run_dir=root/'full',
                               device='cpu',batch_size=1,epochs=2,channels=2,threads=1,
                               save_every=1,deterministic=True,log_samples=0, validation_augmented=True,
                               font_size_min=32, font_size_max=40, padding_min=2, padding_max=6)
            with redirect_stdout(io.StringIO()):train(config)
            interrupted=replace(config,run_dir=root/'interrupted',max_steps=2)
            with redirect_stdout(io.StringIO()):train(interrupted)
            checkpoint=torch.load(interrupted.run_dir/'latest.pt',weights_only=True)
            self.assertIsNotNone(checkpoint['best'])
            # best.pt is a derived artifact and can be repaired from a committed snapshot.
            (interrupted.run_dir/'best.pt').write_bytes(b'broken export')
            with redirect_stdout(io.StringIO()):train(replace(interrupted,resume=True,max_steps=2))
            repaired=torch.load(interrupted.run_dir/'best.pt',weights_only=True)
            self.assert_nested_equal(checkpoint['best'],repaired)
            with redirect_stdout(io.StringIO()):train(replace(interrupted,resume=True,max_steps=None))
            full=torch.load(config.run_dir/'latest.pt',weights_only=True)
            resumed=torch.load(interrupted.run_dir/'latest.pt',weights_only=True)
            for key in ('model','optimizer','scheduler','rng','position'):
                self.assert_nested_equal(full[key],resumed[key])
            for folder in (config.run_dir,interrupted.run_dir):
                events=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
                evaluations=[e for e in events if e['kind']=='validation']
                self.assertEqual(len(evaluations),2)
                best=torch.load(folder/'best.pt',weights_only=True)
                self.assertEqual(best['metrics']['cer'],min(e['cer'] for e in evaluations))
                # Tie retains the first best epoch; the exported weights match the checkpoint.
                state=torch.load(folder/'latest.pt',weights_only=True)
                self.assert_nested_equal(best['model'],state['best']['model'])
            self.assert_nested_equal(full['best']['model'],resumed['best']['model'])
            (interrupted.run_dir/'latest.pt').write_bytes(b'broken checkpoint')
            with redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
                train(replace(interrupted,resume=True,max_steps=None))
            self.assertTrue(caught)
            recovered=torch.load(interrupted.run_dir/'latest.pt',weights_only=True)
            self.assert_nested_equal(full['model'],recovered['model'])
            self.assert_nested_equal(full['best']['model'],recovered['best']['model'])
            events=[json.loads(line) for line in (interrupted.run_dir/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['kind']=='validation' for e in events),2)

            (root/'val.txt').write_text('日本語\n')
            with self.assertRaises(ValueError):train(replace(config,run_dir=root/'overlap'))
