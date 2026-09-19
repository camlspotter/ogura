from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from ogura.training.aliases import CharacterAliases
from ogura.training.metrics import batch_totals, worst_samples, edit_distance
from ogura.training.render import Vocabulary
from ogura.training.train import TrainConfig, train
from tests.test_training import FONT

ALIASES = Path(__file__).resolve().parents[1]/'config/evaluation_aliases.json'


class EvaluationAliasTests(unittest.TestCase):
    def test_scoring_only(self):
        aliases=CharacterAliases.read(ALIASES)
        refs=['A~B','〜'];preds=['A〜B','~']
        raw=batch_totals(preds,refs,3.5,1)
        scored=batch_totals(preds,refs,3.5,1,aliases)
        self.assertEqual(raw['character_errors'],2)
        self.assertEqual(scored['character_errors'],0)
        self.assertEqual(scored['exact_matches'],2)
        self.assertEqual(raw['loss_sum'],scored['loss_sum'])
        self.assertEqual(refs,['A~B','〜'])
        self.assertEqual(preds,['A〜B','~'])
        self.assertEqual(worst_samples(refs,preds,1,aliases)[0][2],0)
        model_aliases=CharacterAliases.read(ALIASES.with_name('character_aliases.json'))
        self.assertEqual(model_aliases.normalize('〜'),'〜')
        v=Vocabulary('~〜',model_aliases.config)
        self.assertEqual(len(v),3)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_training_weights_and_classes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);text=root/'train.txt';text.write_text('A~\nA〜\n')
            val=root/'validation.txt';val.write_text('~A\n')
            vocab=root/'targets.jsonl';vocab.write_text(''.join(json.dumps(dict(character=c))+'\n' for c in ' A~〜'))
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,run_dir=root/'raw',channels=2,
                threads=1,batch_size=1,epochs=1,log_samples=0,device='cpu',validation_text=val)
            train(cfg)
            train(replace(cfg,run_dir=root/'scored',evaluation_aliases=ALIASES))
            raw=torch.load(root/'raw/latest.pt',weights_only=True)
            scored=torch.load(root/'scored/latest.pt',weights_only=True)
            self.assertEqual(raw['model'].keys(),scored['model'].keys())
            for key in raw['model']:self.assertTrue(torch.equal(raw['model'][key],scored['model'][key]))
            self.assertEqual(scored['identity']['evaluation_aliases'],CharacterAliases.read(ALIASES).config)
            best=torch.load(root/'scored/best.pt',weights_only=True)
            self.assertEqual(best['characters'],list(' A~〜'))
