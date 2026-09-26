import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from ogura.textrec.build_english import sha
from ogura.textrec.build_latin_validation import build, eligible
from ogura.textrec.collect_latin_sequences import training_article
from ogura.textrec.evaluate_latin_sequences import sequence_counts, rates


class SequenceMetricsTests(unittest.TestCase):
    def test_denominators_include_correct_examples_and_overlapping_patterns(self):
        counts,_=sequence_counts('office staff skiing','offce staf sking')
        self.assertEqual(counts['ff']['occurrences'],2)
        self.assertEqual(counts['ff']['deleted_occurrences'],1)
        self.assertEqual(counts['ffi']['deleted_occurrences'],1)
        self.assertEqual(counts['fi']['deleted_occurrences'],1)
        self.assertEqual(counts['ii']['deleted_occurrences'],1)
        good,_=sequence_counts('office','office')
        counts['ff'].update(good['ff'])
        self.assertAlmostEqual(rates(counts)['ff']['deletion_rate'],1/3)
        self.assertIsNone(rates(counts)['ij']['deletion_rate'])
        caps,_=sequence_counts('II FF','I F')
        self.assertEqual(caps['ii']['occurrences'],0)
        sub,_=sequence_counts('office','offlce')
        self.assertEqual(sub['ffi']['deleted_occurrences'],0)
        self.assertEqual(sub['ffi']['substituted_occurrences'],1)

    def test_fixed_images_comparison_smoke(self):
        import sys
        import torch
        from ogura.textrec.evaluate_latin_sequences import main
        from ogura.textrec.training.train import EpochDataset,TrainConfig
        from ogura.textrec.training.model import make_model
        from ogura.textrec.training.render import Vocabulary
        from tests.test_training import FONT
        if not FONT.exists():self.skipTest('Font required')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[root/'best.pt',root/'latest.pt']
            for path in paths:path.write_text('fixture checkpoint')
            text=root/'validation.txt';text.write_text('office staff skiing\n')
            (root/'manifest.json').write_text('{}')
            vocab=Vocabulary(sorted(set('office staff skiing')))
            config=TrainConfig(font=FONT)
            dataset=EpochDataset([('office staff skiing','sample')],[0],config,epoch=0)
            model=make_model(len(vocab),2,'small')
            state=dict(identity={},channels=2,model_type='small',model=model.state_dict(),epoch=0,step=0)
            args=['evaluate','--checkpoint',str(paths[0]),'--checkpoint',str(paths[1]),'--validation-text',str(text),'--output',str(root/'report')]
            with patch.object(sys,'argv',args),patch('ogura.textrec.evaluate_latin_sequences.load_validation_context',return_value=(state,vocab,config)),patch('ogura.textrec.evaluate_latin_sequences.validation_sets',return_value=[({'mode':m},dataset) for m in ('baseline','augmented')]):
                main()
            report=json.loads((root/'report/report.json').read_text())
            self.assertEqual(report['checkpoints'][0]['results'],report['checkpoints'][1]['results'])
            rows=[json.loads(s) for s in (root/'report/predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]['render_params'],rows[1]['render_params'])
            self.assertEqual(rows[0]['image_width'],rows[1]['image_width'])
            self.assertEqual(report['checkpoints'][0]['results'][0]['sequences']['ff']['occurrences'],2)


class LatinValidationBuildTests(unittest.TestCase):
    def test_validation_only_provenance_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir()
            articles=[dict(id=str(i),title=f'Article {i}',url=f'https://en.wikipedia.org/wiki/{i}',text=f'The office staff offered {i:05d} different fluffy flowers to the visitors and discussed the effects of the official policy.') for i in range(300)]
            source=root/'source.parquet';pq.write_table(pa.Table.from_pylist(articles),source)
            (base/'train.txt').write_text('unrelated training example\n')
            (base/'targets.jsonl').write_text(''.join(json.dumps({'character':c})+'\n' for c in sorted(set(''.join(a['text'] for a in articles)))))
            (base/'manifest.json').write_text(json.dumps(dict(training_text_sha256=sha(base/'train.txt'),targets_sha256=sha(base/'targets.jsonl'))))
            aliases=root/'aliases.json';aliases.write_text('{"version":1,"groups":[]}')
            held=next(a for a in articles if eligible(a['id']))
            excluded=root/'excluded.jsonl';excluded.write_text(json.dumps(dict(text='Some other held-out text',article_id=held['id'],language='en'))+'\n')
            def args(name):return argparse.Namespace(base=base,output=root/name,corpus=root,aliases=aliases,goal=2,exclude_jsonl=[excluded])
            with patch('ogura.textrec.build_latin_validation.download',return_value=(source,'fixture')):
                build(args('one'));build(args('two'))
            self.assertEqual((root/'one/validation.jsonl').read_bytes(),(root/'two/validation.jsonl').read_bytes())
            rows=[json.loads(s) for s in (root/'one/validation.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            for row in rows:
                self.assertTrue(eligible(row['article_id']))
                self.assertFalse(training_article(row['article_id']))
                self.assertNotEqual(row['article_id'],held['id'])
                self.assertEqual(articles[int(row['article_id'])]['text'][row['start']:row['end']],row['text'])
            manifest=json.loads((root/'one/manifest.json').read_text())
            self.assertEqual(manifest['patterns']['ii']['shortfall'],2)
            with self.assertRaises(FileExistsError):build(args('one'))
