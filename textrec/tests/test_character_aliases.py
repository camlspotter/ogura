from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from ogura.textrec.training.aliases import CharacterAliases
from ogura.textrec.training.render import Vocabulary, BatchRenderer, Sample, RenderParams
from ogura.textrec.training.train import TrainConfig, train, load_initial_weights, sha256
from ogura.textrec.training.model import make_model
from ogura.textrec.training.checkpoint import IdentityMismatch
from tests.test_training import FONT

ALIASES = {'version': 1, 'groups': [{'representative': 'A', 'members': ['A', 'Α', 'А']}]}

class AliasTests(unittest.TestCase):
    def test_validation_and_canonical_mapping(self):
        a = CharacterAliases(ALIASES)
        self.assertEqual(a.normalize('AΑА 0O'), 'AAA 0O')
        for groups in (
            [{'representative': 'B', 'members': ['A', 'Α']}],
            [{'representative': 'A', 'members': ['A', 'A']}],
            ALIASES['groups'] * 2,
            [{'representative': 'A', 'members': ['A', 'ab']}],
            [{'representative': 'A', 'members': ['A', ' ']}],
        ):
            with self.assertRaises(ValueError): CharacterAliases(dict(version=1, groups=groups))
        with self.assertRaises(ValueError): Vocabulary('Α', ALIASES)
        self.assertEqual(Vocabulary('A', ALIASES).characters, ('A',))
        v = Vocabulary(' AΑА0O', ALIASES)
        self.assertEqual(v.characters, tuple(' A0O'))
        self.assertEqual(v.encode('AΑА'), [2, 2, 2])

    def test_fullwidth_configuration_and_unused_members(self):
        from string import ascii_letters
        aliases = CharacterAliases.read(Path(__file__).resolve().parents[1]/'config/character_aliases.json')
        fullwidth = ''.join(chr(ord(c)+0xfee0) for c in ascii_letters)
        self.assertEqual(aliases.normalize(fullwidth), ascii_letters)
        # Future fullwidth corpus entries share classes; no new classes today.
        current = Vocabulary(ascii_letters, aliases.config)
        future = Vocabulary(ascii_letters + fullwidth, aliases.config)
        self.assertEqual(current.characters, future.characters)
        self.assertEqual(future.encode(fullwidth), future.encode(ascii_letters))
        self.assertEqual(aliases.normalize('0０1１ OＯ oｏ'), '0011 OO oo')
        self.assertEqual(aliases.normalize('αβρ'), 'αβρ')
        self.assertEqual(aliases.normalize('０１２３４５６７８９'), '0123456789')
        self.assertEqual(aliases.normalize('0O1Il'), '0O1Il')
        digits = Vocabulary('0123456789０１２３４５６７８９', aliases.config)
        self.assertEqual(digits.characters, tuple('0123456789'))

    def test_symbols_kana_spaces_and_compatibility_kanji(self):
        aliases = CharacterAliases.read(Path(__file__).resolve().parents[1]/'config/character_aliases.json')
        for source, expected in [
            ('Ａ！（Ｂ）％', 'A!(B)%'),
            ('ｱｲｳ ｶﾞﾊﾟｳﾞ ﾜﾞｦﾞ', 'アイウ ガパヴ ヷヺ'),
            ('｢ｶﾀｶﾅ｣･｡､ｰ', '「カタカナ」・。、ー'),
            ('　Ａ 　Ｂ　', ' A B '),
            ('淚淚涙 神神', '淚淚涙 神神'),
            ('ﾞﾟ ｱﾞ −ー一', '゛゜ ア゛ −ー一'),
        ]:
            self.assertEqual(aliases.normalize(source), expected)
            self.assertEqual(aliases.normalize(expected), expected)
        self.assertEqual(CharacterAliases().normalize('　 ｶﾞ'), '　 ｶﾞ')
        with self.assertRaises(ValueError):
            CharacterAliases({'version': 1, 'groups': [{'representative':' ', 'members':[' ', '\n']}]})
        # Adjacent vocabulary entries must not be composed into one character.
        v = Vocabulary('カ゛ガ', aliases.config)
        self.assertEqual(v.characters, tuple('カ゛ガ'))
        self.assertEqual(v.encode('カ゛'), v.encode('ガ'))
        with self.assertRaises(ValueError): Vocabulary('カ゛', aliases.config)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_composed_labels_preserve_original_pixels_and_decode(self):
        from ogura.textrec.training.render import render_sample
        from ogura.textrec.training.metrics import decode
        aliases = CharacterAliases.read(Path(__file__).resolve().parents[1]/'config/character_aliases.json')
        v = Vocabulary(' ガカ゛ｶﾞ　ＡA！!淚淚', aliases.config)
        sample = Sample('ｶﾞ　 Ａ！淚', RenderParams(str(FONT)))
        batch = BatchRenderer(v)([sample])
        original = render_sample(sample)
        pixels = torch.frombuffer(bytearray(original.tobytes()), dtype=torch.uint8).reshape(48,original.width).float()/255
        self.assertTrue(torch.equal(batch.images[0,0,:,:original.width], pixels))
        self.assertEqual(batch.texts, ['ガ A!淚'])
        self.assertEqual(batch.target_lengths.tolist(), [5])
        tokens = [v.ids['カ'], v.ids['゛'], v.ids[' '], 0, v.ids[' '], v.ids['A']]
        logits = torch.full((len(tokens),1,len(v)), -10.)
        for i, token in enumerate(tokens): logits[i,0,token] = 10.
        self.assertEqual(decode(logits,torch.tensor([len(tokens)]),v), ['ガ A'])

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_render_original_and_ctc_repetitions(self):
        v = Vocabulary(' AΑА', ALIASES)
        sample = Sample('AΑА', RenderParams(str(FONT)))
        # Force a narrow original image: mapped AAA needs five CTC positions.
        with patch('ogura.textrec.training.render.render_sample', return_value=Image.new('L', (24,48),255)) as render:
            batch = BatchRenderer(v)([sample])
        self.assertEqual(render.call_args.args[0].text, 'AΑА')
        self.assertEqual(sample.text, 'AΑА')
        self.assertEqual(batch.texts, ['AAA'])
        self.assertEqual(batch.targets.tolist(), [2,2,2])
        self.assertGreaterEqual((batch.image_widths.item()+7)//8, 5)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_checkpoint_resume_and_export(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp)
            text=root/'train.txt';text.write_text('AΑА\nΑАA\n')
            val=root/'validation.txt';val.write_text('АAΑA\n')
            vocab=root/'targets.jsonl';vocab.write_text(''.join(json.dumps({'character':c})+'\n' for c in ' AΑА'))
            aliases=root/'aliases.json';aliases.write_text(json.dumps({**ALIASES, 'version':2, 'collapse_ascii_spaces':True, 'compose_katakana_diacritics':True}))
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,character_aliases=aliases,
                run_dir=root/'run',channels=2,threads=1,batch_size=1,epochs=1,log_samples=0,
                device='cpu',validation_text=val)
            train(replace(cfg,max_steps=1))
            train(replace(cfg,resume=True))
            state=torch.load(root/'run/best.pt',weights_only=True)
            restored=Vocabulary(state['source_characters'],state['identity']['character_aliases'])
            self.assertEqual(list(restored.characters),state['characters'])
            self.assertEqual(restored.aliases.normalize('ΑА'), 'AA')
            # Evaluation must restore aliases from the checkpoint without the config file.
            (root/'manifest.json').write_text(json.dumps(dict(
                training_text_sha256=sha256(text), targets_sha256=sha256(vocab),
                validation_text_sha256=sha256(val), split='validation',
                min_length=4, max_length=4, samples=1)))
            aliases.rename(root/'moved-aliases.json')
            from ogura.textrec.evaluate_lengths import main as evaluate_main
            with patch('sys.argv', ['evaluate', '--checkpoint', str(root/'run/best.pt'),
                        '--device', 'cpu', '--validation-text', str(val),
                        '--output', str(root/'evaluation.json')]):
                evaluate_main()
            aliases = root/'moved-aliases.json'
            cfg = replace(cfg, character_aliases=aliases)
            model=make_model(len(restored),2)
            load_initial_weights(model,root/'run/best.pt',restored,2)
            with self.assertRaises(ValueError):
                load_initial_weights(model,root/'run/best.pt',Vocabulary(' A'),2)
            with self.assertRaises(IdentityMismatch):
                train(replace(cfg,resume=True,character_aliases=None))
