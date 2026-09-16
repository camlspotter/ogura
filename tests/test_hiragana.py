import json
from pathlib import Path
import tempfile
import unittest
from ogura.generate_hiragana import generate, HIRAGANA
from ogura.text_common import digest


class HiraganaTests(unittest.TestCase):
    def test_deficits_provenance_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/'base'; base.mkdir()
            text='これは漢字について説明するための文章です'
            (base/'train.jsonl').write_text(json.dumps(dict(text=text,sample_id=digest(text),length=len(text)),ensure_ascii=False)+'\n')
            targets=sorted(set(text+HIRAGANA+'寙'))
            (base/'targets.jsonl').write_text(''.join(json.dumps(dict(character=c),ensure_ascii=False)+'\n' for c in targets))
            for name in ('a','b'):generate(base,root/name,goal=2,seed=7)
            self.assertEqual((root/'a/train.txt').read_bytes(),(root/'b/train.txt').read_bytes())
            coverage={r['character']:r for r in map(json.loads,(root/'a/coverage.jsonl').read_text().splitlines())}
            self.assertEqual(coverage['寙']['natural_count'],0)
            self.assertEqual(coverage['寙']['generated_for_character'],2)
            self.assertEqual(coverage['漢']['generated_for_character'],1)
            rows=list(map(json.loads,(root/'a/train.jsonl').read_text().splitlines()))
            self.assertIn(text,[r['text'] for r in rows if not r.get('synthetic')])
            for r in rows:
                if r.get('synthetic'):
                    self.assertNotIn('article_id',r)
                    self.assertLessEqual(len(r['embedded_targets']),5)
                    for entry in r['embedded_targets']:
                        self.assertEqual(r['text'].count(entry['character']),1)
            self.assertTrue(any(len(r.get('embedded_targets',[])) == 5 for r in rows))
