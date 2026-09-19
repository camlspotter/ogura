from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ogura.diagnose_validation import align_errors, diagnose, main
from ogura.training.metrics import edit_distance
from ogura.training.render import Sample, RenderParams, Vocabulary
from tests.test_training import FONT


class DiagnosticsTests(unittest.TestCase):
    def test_alignment_and_positions(self):
        texts=['']+[''.join(chars) for n in (1,2,3) for chars in itertools.product('AB',repeat=n)]
        for ref in texts:
            for pred in texts:
                edits=align_errors(ref,pred)
                self.assertEqual(len(edits),edit_distance(ref,pred))
                for e in edits:
                    if e['reference']:self.assertEqual(ref[e['reference_index']],e['reference'])
                    if e['prediction']:self.assertEqual(pred[e['prediction_index']],e['prediction'])
        self.assertEqual(align_errors('ABC','AXC')[0]['kind'],'substitution')
        self.assertEqual(align_errors('ABC','AC')[0]['reference_index'],1)
        self.assertEqual(align_errors('AC','ABC')[0]['reference_index'],1)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_all_counts_top_images_and_html_escape(self):
        vocabulary=Vocabulary(' AB')
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.index=0
            def output_lengths(self,widths):return torch.full_like(widths,2)
            def forward(self,images):
                predictions=['BA','A','AB'][self.index:self.index+len(images)]
                self.index+=len(images)
                logits=torch.full((2,len(images),len(vocabulary)),-10.)
                for b,text in enumerate(predictions):
                    for t in range(2):logits[t,b,vocabulary.ids[text[t]] if t<len(text) else 0]=10
                return logits
        dataset=[Sample(text,RenderParams(str(FONT)),str(i)) for i,text in enumerate(['AB','AA','AB'])]
        info=dict(dataset='<unsafe>',mode='baseline',min_length=2,max_length=2)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            out=Path(tmp)/'report'
            rows=diagnose(Model(),[(info,dataset)],vocabulary,2,torch.device('cpu'),out,1)
            row=rows[0]
            self.assertEqual(row['reference_occurrences'],{'A':4,'B':2})
            self.assertEqual(row['character_errors'],3)
            self.assertEqual(row['cer'],0.5)
            self.assertEqual(row['accuracy'],1/3)
            self.assertEqual(len(list((out/'images').glob('*.png'))),1)
            self.assertEqual(row['worst'][0]['sample_index'],0)
            self.assertEqual(len((out/'errors.jsonl').read_text().splitlines()),2)
            self.assertIn('&lt;unsafe&gt;', (out/'index.html').read_text())
            self.assertNotIn('<unsafe>', (out/'index.html').read_text())
            with self.assertRaises(FileExistsError):diagnose(Model(),[(info,dataset)],vocabulary,2,torch.device('cpu'),out,1)

    @unittest.skipUnless(FONT.exists(), 'Noto font required')
    def test_cli_real_checkpoint_and_validation_manifest(self):
        from ogura.training.train import TrainConfig, identity_for, sha256
        from ogura.training.model import make_model
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);text=root/'train.txt';text.write_text('AB\n')
            val=root/'validation.txt';val.write_text('BAAAA\n')
            vocab=root/'targets.jsonl';vocab.write_text(''.join(json.dumps(dict(character=c))+'\n' for c in ' AB'))
            cfg=TrainConfig(text=text,vocabulary=vocab,font=FONT,channels=2,validation_text=val)
            identity=identity_for(cfg,torch.device('cpu'))
            (root/'manifest.json').write_text(json.dumps(dict(training_text_sha256=sha256(text),targets_sha256=sha256(vocab),validation_text_sha256=sha256(val),split='validation',samples=1,min_length=5,max_length=5)))
            (root/'run_config.json').write_text(json.dumps(dict(arguments=dict(font=str(FONT)))))
            model=make_model(4,2)
            with torch.no_grad():
                model.classifier.weight.zero_();model.classifier.bias.zero_();model.classifier.bias[0]=10
            checkpoint=root/'best.pt'
            torch.save(dict(identity=identity,characters=list(' AB'),channels=2,model=model.state_dict(),epoch=1,step=1),checkpoint)
            out=root/'diagnostics'
            with patch('sys.argv',['diagnose','--checkpoint',str(checkpoint),'--device','cpu','--validation-text',str(val),'--output',str(out),'--top','1']):main()
            rows=json.loads((out/'conditions.json').read_text())
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['cer']==1 for r in rows))
            self.assertEqual(json.loads((out/'manifest.json').read_text())['checkpoint_sha256'],sha256(checkpoint))
