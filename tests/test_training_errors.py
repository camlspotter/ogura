import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ogura.training.error_samples import error_rows
from ogura.training.render import Sample,RenderParams
from ogura.training.train import TrainConfig,train
from tests.test_training import FONT


class ErrorSamplesTests(unittest.TestCase):
    def test_only_worst_three_errors_and_resolved_inputs(self):
        samples=[Sample('raw'+str(i),RenderParams('font',font_size=28),str(i)) for i in range(5)]
        rows=error_rows(samples,['abc']*5,['abc','a','x','abcdef',''],[100]*5,2,100,200)
        self.assertEqual([r['batch_sample_index'] for r in rows],[2,3,4])
        self.assertEqual(rows[0]['input_text'],'raw2')
        self.assertEqual(rows[0]['render_params']['font_size'],28)
        self.assertEqual(error_rows(samples[:1],['a'],['a'],[10],1,100,100),[])

    @unittest.skipUnless(FONT.exists(),'Noto font required')
    def test_checkpoint_offset_and_resume_truncates_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);text=root/'train.txt';text.write_text('日本語\n'*101)
            vocab=root/'targets.jsonl';vocab.write_text(''.join(json.dumps({'character':c})+'\n' for c in ' 日本語'))
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,run_dir=root/'run',device='cpu',
                            channels=2,threads=1,batch_size=1,epochs=1,max_steps=100,save_every=100,log_every=100,log_samples=0)
            with patch('ogura.training.train.decode',return_value=['日']):train(cfg)
            log=root/'run/training_errors.jsonl';before=log.read_bytes()
            rows=[json.loads(s) for s in before.splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['batch'],100)
            self.assertEqual(rows[0]['reference'],'日本語')
            ckpt=torch.load(root/'run/latest.pt',weights_only=True)
            self.assertEqual(ckpt['metrics']['error_log_offset'],len(before))
            with log.open('ab') as f:f.write(b'{"uncommitted":true}\n')
            with patch('ogura.training.train.decode',return_value=['日']):train(replace(cfg,resume=True,max_steps=101))
            self.assertEqual(log.read_bytes(),before)
            self.assertTrue((root/'run/training_errors_metadata.json').exists())
