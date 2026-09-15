import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from synthesize_shortfalls import apply_replacements
from extract_training_text import digest


class SynthesisTest(unittest.TestCase):
    def test_classes_and_provenance(self):
        self.assertEqual(apply_replacements('漢字®',[{'position':0,'from':'漢','to':'寙'},{'position':2,'from':'®','to':'〄'}]),'寙字〄')
        for edit in [{'position':0,'from':'漢','to':'〄'},{'position':0,'from':'字','to':'寙'}]:
            with self.assertRaises(ValueError):apply_replacements('漢字',[edit])

    def test_repeatable_small_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir()
            texts=['漢字©'+'漢'*17,'字漢®'+'漢'*17]; chars='漢字©®寙〄'
            rows=[{'text':t,'sample_id':digest(t),'length':len(t),'article_id':'1','url':'test','title':'test','start':0,'end':len(t),'anchor':'漢'} for t in texts]
            (base/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            (base/'targets.jsonl').write_text(''.join(json.dumps({'character':c,'ranking_occurrences':1},ensure_ascii=False)+'\n' for c in chars))
            (base/'coverage.jsonl').write_text(''.join(json.dumps({'character':c,'sample_count':sum(c in t for t in texts)},ensure_ascii=False)+'\n' for c in chars))
            (base/'manifest.json').write_text(json.dumps({'seed':20260915,'min_tokens':20}))
            (base/'summary.json').write_text(json.dumps({'samples':2,'articles':1,'article_splits':{'train':1},'min_tokens':20}))
            (base/'article_splits.jsonl').write_text('')
            for name in ['a','b']:
                subprocess.run([sys.executable,str(Path(__file__).with_name('synthesize_shortfalls.py')),'--base',str(base),'--output',str(root/name),'--report-dir',str(root/(name+'report')),'--goal','2','--seed','7'],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual((root/'a/train.jsonl').read_bytes(),(root/'b/train.jsonl').read_bytes())
            coverage=[json.loads(line) for line in (root/'a/coverage.jsonl').read_text().split('\n') if line]
            self.assertTrue(all(r['sample_count']>=2 for r in coverage))
            for r in (json.loads(line) for line in (root/'a/synthetic.jsonl').read_text().split('\n') if line):
                self.assertEqual(apply_replacements(r['base_text'],r['replacements']),r['text'])
                self.assertEqual(len(r['text']),20)


if __name__=='__main__':unittest.main()
