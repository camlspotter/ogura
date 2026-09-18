from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from PIL import ImageChops

from ogura.training.render import (RenderParams, Sample, BatchRenderer, Vocabulary,
    normalized_text, western_runs, western_positions, render_sample, parameters_for_sample)
from ogura.training.model import make_model, ctc_loss
from ogura.training.train import TrainConfig, train, sha256, identity_for
from ogura.training.checkpoint import IdentityMismatch
from ogura.evaluate_lengths import validation_sets, validation_font_keys, main as evaluate_main
from tests.test_training import FONT

T = FONT.with_name('Tinos-Regular.ttf')
A = FONT.with_name('Arimo[wght].ttf')


@unittest.skipUnless(all(p.exists() for p in (FONT,T,A)), 'Downloaded mixed fonts required')
class MixedTests(unittest.TestCase):
    def test_routing_fallback_and_preserved_fullwidth(self):
        text='日本 “ISO 123 αβ АБ” Ａ１２'
        runs=western_runs(text,str(FONT),str(T))
        mapping=[(c,p) for run,p in runs for c in run]
        for c,p in mapping:
            if c in '日本Ａ１２':self.assertEqual(p,str(FONT))
            if c in 'ISO123αβАБ“”':self.assertEqual(p,str(T))
        p=RenderParams(str(FONT),western_font_path=str(T))
        self.assertEqual(normalized_text(text,p),text)
        with patch('ogura.training.render.font_characters',
                   side_effect=lambda path:frozenset(map(ord,' 日Ω' if path=='jp' else ' A'))):
            self.assertEqual(western_runs('AΩ','jp','en'),[('A','en'),('Ω','jp')])
            self.assertEqual(normalized_text('AΩ☃☃日',RenderParams('jp',western_font_path='en')),'AΩ 日')

    def test_kerning_target_pixels_and_ctc_repeats(self):
        torch.set_num_threads(1)
        for f in (T,A):
            pair=western_positions('AV',str(f),40)[1]
            individual=sum(western_positions(c,str(f),40)[1] for c in 'AV')
            self.assertLess(pair,individual)
        p=RenderParams(str(FONT),western_font_path=str(T))
        text='日本 Mill αβ АБ Ａ１２\U0010ffff\U0010fffe'
        norm=normalized_text(text,p)
        self.assertEqual(norm,text[:-2]+' ')
        self.assertIsNone(ImageChops.difference(render_sample(Sample(text,p)),
                                              render_sample(Sample(norm,p))).getbbox())
        narrow='i'*25
        batch=BatchRenderer(Vocabulary('i'))([Sample(narrow,p)])
        self.assertEqual(batch.texts,[narrow])
        self.assertGreaterEqual(int(make_model(2,2).output_lengths(batch.image_widths)[0]),49)
        self.assertTrue(torch.isfinite(ctc_loss(make_model(2,2)(batch.images),batch)))
        for y in (0.,1.):
            im=render_sample(Sample('日本 gj αβ',replace(p,vertical_position=y)))
            box=ImageChops.invert(im).getbbox()
            self.assertGreaterEqual(box[1],p.padding)
            self.assertLessEqual(box[3],48-p.padding)

    def test_sampler_reproducibility(self):
        args=(FONT,12,3,'test',28,40)
        plain=parameters_for_sample(*args)
        mixed=parameters_for_sample(*args,western_fonts=(T,A))
        self.assertEqual(replace(mixed,western_font_path=None),plain)
        self.assertEqual(mixed,parameters_for_sample(*args,western_fonts=(T,A)))
        self.assertIsNone(parameters_for_sample(*args,clean_probability=1,western_fonts=(T,A)).western_font_path)

    def test_training_resume_grid_and_standalone_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            text=root/'train.txt';text.write_text('日本 Mill\n語 αβ АБ\n')
            chars=' 日本語文字MillαβАБ'
            vocab=root/'targets.jsonl'
            vocab.write_text(''.join(json.dumps(dict(character=c))+'\n' for c in dict.fromkeys(chars)))
            paths=[]
            for n in (5,25,80):
                d=root/str(n);d.mkdir()
                p=d/'validation.txt';p.write_text(('文Mill'*16)[:n]+'\n')
                (d/'targets.jsonl').write_bytes(vocab.read_bytes())
                (d/'manifest.json').write_text(json.dumps(dict(samples=1,min_length=n,max_length=n,
                   split='validation',training_text_sha256=sha256(text),targets_sha256=sha256(vocab),
                   validation_text_sha256=sha256(p))))
                paths.append(p)
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,western_fonts=(T,A),run_dir=root/'full',
                channels=2,threads=1,batch_size=1,epochs=1,log_samples=0,device='cpu',
                validation_text=paths[1],validation_augmented=True,monitor_validation=(paths[0],paths[2]),
                selection_metric='mean-font-cer',clean_probability=0)
            identity=identity_for(cfg,torch.device('cpu'))
            self.assertEqual(len(validation_font_keys(identity)),2)
            datasets=validation_sets(cfg,Vocabulary.read(vocab),paths,identity,all_fonts=True)
            self.assertEqual(len(datasets),9)
            for info,ds in datasets:
                sample=ds[0]
                if info['mode']=='baseline':self.assertIsNone(sample.render_params.western_font_path)
                else:self.assertIn(Path(sample.render_params.western_font_path).name,info['font'])
            with redirect_stdout(io.StringIO()):
                train(cfg)
                train(replace(cfg,run_dir=root/'resumed',max_steps=1))
                train(replace(cfg,run_dir=root/'resumed',resume=True))
            full=torch.load(root/'full/latest.pt',weights_only=True)
            resumed=torch.load(root/'resumed/latest.pt',weights_only=True)
            for k,v in full['model'].items():self.assertTrue(torch.equal(v,resumed['model'][k]))
            with self.assertRaises(IdentityMismatch):
                train(replace(cfg,run_dir=root/'resumed',resume=True,western_fonts=(A,T)))
            argv=['evaluate','--checkpoint',str(root/'full/best.pt'),'--device','cpu',
                  '--output',str(root/'report.json')]
            for p in paths:argv+=['--validation-text',str(p)]
            with patch('sys.argv',argv),redirect_stdout(io.StringIO()):evaluate_main()
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['grid']['overall']['conditions'],6)
