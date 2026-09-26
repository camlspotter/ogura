import unittest
from ogura.textrec.evaluate_real import score
from ogura.textrec.training.aliases import CharacterAliases


class RealEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.aliases=CharacterAliases({'version':1,'groups':[{'representative':'A','members':['A','Ａ']}]})
        self.evaluation=CharacterAliases()
        self.refs=[dict(image='one.png',reference='Ａ B',reference_verified=True),
                   dict(image='two.png',reference='ISO',reference_verified=True)]

    def run_score(self, preds, refs=None):
        return score(self.refs if refs is None else refs,preds,self.aliases,self.evaluation)

    def test_normalization_edges_and_micro_cer(self):
        preds=[dict(image='different/path/one.png',preprocessing='contrast',prediction=' A B'),
               dict(image='two.png',preprocessing='contrast',prediction='IS0')]
        result=self.run_score(preds)
        self.assertEqual(result['errors'],2)
        self.assertEqual(result['cer'],2/6)
        self.assertEqual(result['space_errors'],1)
        self.assertEqual(result['exact_matches'],0)
        self.assertEqual(result['rows'][0]['raw_reference'],'Ａ B')

    def test_missing_duplicate_and_unverified_rejected(self):
        pred=dict(image='one.png',preprocessing='contrast',prediction='A B')
        for predictions in [[],[pred],[pred,pred]]:
            with self.assertRaises(ValueError):self.run_score(predictions)
        with self.assertRaisesRegex(ValueError,'Unverified'):
            self.run_score([pred],[dict(self.refs[0],reference_verified=False)])
        with self.assertRaisesRegex(ValueError,'original JSONL'):
            self.run_score([dict(pred,prediction='A\nB')])

    def test_preprocessing_selection(self):
        rows=[dict(image=r['image'],prediction=r['reference'],preprocessing='contrast') for r in self.refs]
        rows.append(dict(image='one.png',prediction='wrong',preprocessing='none'))
        result=self.run_score(rows)
        self.assertEqual(result['cer'],0)
        self.assertEqual(result['accuracy'],1)

    def test_cli_comparison_and_image_hash(self):
        import hashlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from contextlib import redirect_stdout
        from ogura.textrec.evaluate_real import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'one.png').write_bytes(b'image fixture')
            reference=[dict(image='one.png',reference='ABC',reference_verified=True,
                            image_sha256=hashlib.sha256(b'image fixture').hexdigest())]
            refs=root/'refs.jsonl';refs.write_text(json.dumps(reference[0])+'\n')
            old=root/'old.jsonl';new=root/'new.jsonl'
            old.write_text(json.dumps(dict(image='one.png',preprocessing='contrast',prediction='AC'))+'\n')
            new.write_text(json.dumps(dict(image='one.png',preprocessing='contrast',prediction='ABC'))+'\n')
            out=root/'out.json'
            args=['evaluate_real','--references',str(refs),'--output',str(out),str(old),str(new)]
            with patch('sys.argv',args),redirect_stdout(io.StringIO()):main()
            result=json.loads(out.read_text())
            self.assertEqual(result['comparisons'][0]['error_delta'],-1)
            out.unlink();(root/'one.png').write_bytes(b'changed image')
            with patch('sys.argv',args), self.assertRaisesRegex(ValueError,'image changed'):main()
