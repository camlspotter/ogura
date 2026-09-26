from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from ogura.textrec.training.migration import migrate_classifier
from ogura.textrec.training.model import make_model
from ogura.textrec.training.render import Vocabulary
from ogura.textrec.training.train import TrainConfig, train, sha256
from tests.test_character_aliases import ALIASES
from tests.test_training import FONT


class MigrationTests(unittest.TestCase):
    def test_exact_copies_and_mean_for_both_architectures(self):
        for kind in ('small', 'residual'):
            old = Vocabulary(' AΑАB')
            new = Vocabulary(old.source_characters, ALIASES)
            source = make_model(len(old), 2, kind)
            with torch.no_grad():
                for i in range(len(old)):
                    source.classifier.weight[i].fill_(i)
                    source.classifier.bias[i] = i * 2
            state = dict(characters=list(old.characters),channels=2,model_type=kind,model=source.state_dict())
            dest = make_model(len(new), 2, kind)
            report = migrate_classifier(dest,state,new,2,kind)
            self.assertEqual(len(report['merged_groups']),1)
            for key, value in source.state_dict().items():
                if not key.startswith('classifier.'):
                    self.assertTrue(torch.equal(value,dest.state_dict()[key]))
            for c in ' B':
                self.assertTrue(torch.equal(source.classifier.weight[old.ids[c]],dest.classifier.weight[new.ids[c]]))
            self.assertTrue(torch.equal(source.classifier.weight[0], dest.classifier.weight[0]))
            self.assertTrue(torch.equal(source.classifier.bias[0], dest.classifier.bias[0]))
            self.assertTrue(torch.equal(dest.classifier.weight[new.ids['A']], source.classifier.weight[[2,3,4]].mean(0)))
            self.assertEqual(dest.classifier.bias[new.ids['A']].item(), 6.)
            with self.assertRaises(ValueError):migrate_classifier(dest,state,new,3,kind)
            merged_state = dict(state, characters=list(new.characters), source_characters=list(old.characters),
                                identity={'character_aliases':new.aliases.config},model=dest.state_dict())
            with self.assertRaises(ValueError):migrate_classifier(source,merged_state,old,2,kind)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_legacy_latest_validation_zero_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)
            text=root/'train.txt';text.write_text('AΑА\nΑАA\n')
            val=root/'val.txt';val.write_text('BAΑА\n')
            vocab=root/'targets.jsonl';vocab.write_text(''.join(json.dumps({'character':c})+'\n' for c in ' AΑАB'))
            old=Vocabulary.read(vocab)
            source=make_model(len(old),2)
            ckpt=root/'old-latest.pt'
            torch.save(dict(identity={'vocabulary_sha256':sha256(vocab),'settings':{'channels':2}},
                            position={'epoch':16,'step':1200},model=source.state_dict()),ckpt)
            aliases=root/'aliases.json';aliases.write_text(json.dumps(ALIASES))
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,character_aliases=aliases,
                            init_from=ckpt,migrate_aliases=True,run_dir=root/'run',channels=2,
                            threads=1,batch_size=1,epochs=1,log_samples=0,device='cpu',validation_text=val)
            train(replace(cfg,max_steps=1))
            latest=torch.load(root/'run/latest.pt',weights_only=True)
            self.assertEqual(latest['initialization']['source_step'],1200)
            events=[json.loads(line) for line in (root/'run/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(any(e.get('kind')=='validation' and e.get('epoch')==0 for e in events))
            train(replace(cfg,resume=True,init_from=None,migrate_aliases=False))
            resumed=torch.load(root/'run/latest.pt',weights_only=True)
            self.assertEqual(resumed['initialization'],latest['initialization'])
            best=torch.load(root/'run/best.pt',weights_only=True)
            self.assertEqual(best['initialization'],latest['initialization'])
            self.assertEqual(best['characters'],list(Vocabulary.read(vocab,ALIASES).characters))
            wrong=root/'wrong.jsonl';wrong.write_text('{}')
            with self.assertRaises(ValueError):
                migrate_classifier(make_model(4,2),torch.load(ckpt,weights_only=True),Vocabulary.read(vocab,ALIASES),2,'small',wrong)

    def test_add_quotes_and_merge_wave_preserves_old_features(self):
        aliases={'version':1,'groups':[{'representative':'~','members':['~','〜']}]}
        old=Vocabulary(' ~〜A')
        new=Vocabulary('”A〜“~‘ ’',aliases)
        for kind in ('small','residual'):
            source=make_model(len(old),2,kind)
            state=dict(characters=list(old.characters),source_characters=list(old.source_characters),
                       channels=2,model_type=kind,model=source.state_dict())
            dest=make_model(len(new),2,kind)
            with self.assertRaisesRegex(ValueError,'allow-new-classes'):
                migrate_classifier(dest,state,new,2,kind)
            report=migrate_classifier(dest,state,new,2,kind,allow_new_classes=True)
            self.assertEqual(set(report['initialized_classes']),set('“”‘’'))
            for key,value in source.state_dict().items():
                if not key.startswith('classifier.'):
                    self.assertTrue(torch.equal(value,dest.state_dict()[key]))
            for key in ('classifier.weight','classifier.bias'):
                actual=dest.state_dict()[key]; original=source.state_dict()[key]
                self.assertTrue(torch.equal(actual[0],original[0]))
                for c in ' A':self.assertTrue(torch.equal(actual[new.ids[c]],original[old.ids[c]]))
                for c in '“”‘’':
                    expected=original[0] - (5.0 if key.endswith('bias') else 0)
                    self.assertTrue(torch.equal(actual[new.ids[c]],expected))
                self.assertTrue(torch.equal(actual[new.ids['~']],original[[old.ids['~'],old.ids['〜']]].mean(0)))
            with self.assertRaisesRegex(ValueError,'remove'):
                migrate_classifier(dest,state,Vocabulary(' ~A“”‘’'),2,kind,allow_new_classes=True)
            merged=dict(state,characters=list(new.characters),source_characters=list(new.source_characters),
                        identity={'character_aliases':aliases},model=dest.state_dict())
            with self.assertRaisesRegex(ValueError,'split'):
                migrate_classifier(make_model(len(new)+1,2,kind),merged,Vocabulary(new.source_characters),2,kind,allow_new_classes=True)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_training_with_added_classes_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);old=Vocabulary(' A~〜')
            source=make_model(len(old),2)
            checkpoint=root/'old.pt'
            torch.save(dict(characters=list(old.characters),source_characters=list(old.source_characters),
                            channels=2,model=source.state_dict()),checkpoint)
            vocab=root/'targets.jsonl'
            vocab.write_text(''.join(json.dumps({'character':c})+'\n' for c in ' A~〜“”‘’'))
            text=root/'train.txt';text.write_text('“A”\n‘A’\n')
            alias=root/'alias.json';alias.write_text(json.dumps({'version':1,'groups':[{'representative':'~','members':['~','〜']}]}))
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,character_aliases=alias,
                            init_from=checkpoint,migrate_aliases=True,allow_new_classes=True,
                            run_dir=root/'run',channels=2,threads=1,batch_size=1,epochs=1,
                            log_samples=0,device='cpu')
            train(replace(cfg,max_steps=1))
            state=torch.load(root/'run/latest.pt',weights_only=True)
            self.assertEqual(set(state['initialization']['initialized_classes']),set('“”‘’'))
            train(replace(cfg,resume=True,init_from=None,migrate_aliases=False,allow_new_classes=False))
            resumed=torch.load(root/'run/latest.pt',weights_only=True)
            self.assertEqual(state['initialization'],resumed['initialization'])
            self.assertEqual(resumed['position']['step'],2)

    def test_new_quotes_cannot_outrank_donors_and_can_learn(self):
        old=Vocabulary(' A\"\'')
        new=Vocabulary('”A“\"‘ \'’')
        source=make_model(len(old),2)
        state=dict(characters=list(old.characters),source_characters=list(old.source_characters),
                   channels=2,model=source.state_dict())
        dest=make_model(len(new),2)
        report=migrate_classifier(dest,state,new,2,'small',allow_new_classes=True)
        self.assertEqual({r['donor'] for r in report['new_class_initialization']},{'"',"'"})
        features=torch.randn(100,source.classifier.in_features)*100
        logits=dest.classifier(features)
        for c,d in [('“','"'),('”','"'),('‘',"'"),('’',"'")]:
            torch.testing.assert_close(logits[:,new.ids[c]],logits[:,new.ids[d]]-5,atol=1e-4,rtol=1e-5)
        self.assertFalse(any(i in {new.ids[c] for c in '“”‘’'} for i in logits.argmax(-1).tolist()))
        expected=[old.characters[i-1] if i else '<blank>' for i in source.classifier(features).argmax(-1).tolist()]
        actual=[new.characters[i-1] if i else '<blank>' for i in logits.argmax(-1).tolist()]
        self.assertEqual(expected,actual)
        loss=torch.nn.functional.cross_entropy(logits,torch.full((100,),new.ids['“']))
        loss.backward()
        self.assertGreater(dest.classifier.weight.grad[new.ids['“']].abs().sum().item(),0)
