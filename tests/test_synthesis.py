import json
from pathlib import Path
import pickle
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zlib
from ogura.synthesize_shortfalls import apply_replacements
from ogura.text_common import digest


class SynthesisTest(unittest.TestCase):
    def test_classes_and_provenance(self):
        self.assertEqual(apply_replacements('漢字©',[{'position':0,'from':'漢','to':'寙'},{'position':2,'from':'©','to':'〄'}]),'寙字〄')
        for edit in [{'position':0,'from':'漢','to':'〄'},{'position':0,'from':'字','to':'寙'}]:
            with self.assertRaises(ValueError):apply_replacements('漢字',[edit])

    def test_repeatable_distinct_donor_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir();index=root/'index';index.mkdir()
            rng=random.Random(12);alphabet='東京大阪京都北海道日本世界歴史文化美術科学研究';articles=[]
            for i in range(100):
                text=''.join(rng.choices(alphabet,k=8))+'について'+''.join(rng.choices(alphabet,k=7))+('★' if i>=50 else '©')
                self.assertEqual(len(text),20)
                articles.append({'id':str(i),'url':'test','title':'本文','text':text})
            chars=sorted((set(''.join(a['text'] for a in articles))-{'★'})|set('寙〄'))
            rows=[dict(a,sample_id=digest(a['text']),length=20,article_id=a['id'],start=0,end=20,anchor=a['text'][0]) for a in articles[:10]]
            (base/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            (base/'targets.jsonl').write_text(''.join(json.dumps({'character':c,'ranking_occurrences':1},ensure_ascii=False)+'\n' for c in chars))
            (base/'coverage.jsonl').write_text(''.join(json.dumps({'character':c,'sample_count':sum(c in r['text'] for r in rows)},ensure_ascii=False)+'\n' for c in chars))
            meta={'seed':20260915,'min_tokens':20,'source_revision':'fixture'}
            (base/'manifest.json').write_text(json.dumps(meta))
            (base/'summary.json').write_text(json.dumps(dict(meta,samples=10,articles=100,article_splits={'train':100})))
            (base/'article_splits.jsonl').write_text('')
            with sqlite3.connect(index/'articles.sqlite3') as db:
                db.execute('create table articles(n integer primary key,payload blob)')
                db.executemany('insert into articles values(?,?)',[(i,zlib.compress(json.dumps(a).encode())) for i,a in enumerate(articles)])
            with (index/'complete.pkl').open('wb') as f:
                pickle.dump(([[i for i,a in enumerate(articles) if c in a['text']] for c in chars],{}, {},100,chars,'fixture',20260915),f)
            for name in ['a','b']:
                subprocess.run([sys.executable,'-m','ogura.synthesize_shortfalls','--base',str(base),'--donor-index',str(index),'--output',str(root/name),'--report-dir',str(root/(name+'report')),'--goal','2','--seed','7'],cwd=Path(__file__).resolve().parents[1],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual((root/'a/train.jsonl').read_bytes(),(root/'b/train.jsonl').read_bytes())
            coverage=[json.loads(line) for line in (root/'a/coverage.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['sample_count']>=2 for r in coverage))
            donors={r["sample_id"] for r in rows}
            for r in map(json.loads,(root/'a/synthetic.jsonl').read_text().splitlines()):
                self.assertEqual(apply_replacements(r['base_text'],r['replacements']),r['text'])
                self.assertEqual(len(r['text']),20)
                self.assertTrue(set(r['text'])<=set(chars))
                self.assertNotIn(r['base_sample_id'],donors);donors.add(r['base_sample_id'])


if __name__=='__main__':unittest.main()
