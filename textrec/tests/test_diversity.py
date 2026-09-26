import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from ogura.textrec.diversity import DiversityFilter
from ogura.textrec.prose_filter import is_prose
from ogura.textrec.synthesize_shortfalls import apply_replacements


class DiversityTests(unittest.TestCase):
    def test_source_overlap_after_other_articles(self):
        f=DiversityFilter();f.add('a',0,25,'東京の博物館では歴史に関する資料を展示しています。')
        f.add('b',0,25,'大阪の公園には多くの樹木が植えられているそうです。')
        self.assertFalse(f.allows('a',10,35,'別の文字列でも原文区間の重なりは除外する。'))
        self.assertTrue(f.allows('a',25,50,'北海道では新しい鉄道の計画が発表されたようです。'))

    def test_shifted_duplicate_across_articles(self):
        f=DiversityFilter();f.add('a',0,25,'これは同じ原文から少しずつ切り出す例です。')
        self.assertFalse(f.allows('b',8,33,'同じ原文から少しずつ切り出す例です。その'))

    def test_character_tables(self):
        self.assertFalse(is_prose('ゅょゎゕゖㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇷ゚ㇺㇻㇼㇽㇾㇿ々'))
        self.assertFalse(is_prose('𫵷 岠 岜 呇 冏 觃 岙 伾 㑇 伭 佖 伲'))
        self.assertFalse(is_prose('𫵷 岠 岜 呇 冏 觃 の文字を一覧として並べています。'))
        self.assertTrue(is_prose('東京の博物館では歴史に関する資料を展示しています。'))
        self.assertTrue(is_prose('この製品には工業規格を示す©の記号が付いています。'))

    def test_shuffle_is_repeatable_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs=[Path(tmp)/n for n in ('a','b')]
            original=[json.dumps({'id':n})+'\n' for n in range(20)]
            for d in dirs:
                d.mkdir();(d/'train.jsonl').write_text(''.join(original))
                for name in ('manifest.json','summary.json'):(d/name).write_text('{}')
                subprocess.run([sys.executable,'-m','ogura.textrec.shuffle_training_text','--output',str(d),'--seed','12'],cwd=Path(__file__).resolve().parents[1],check=True)
            a=(dirs[0]/'train.jsonl').read_text();b=(dirs[1]/'train.jsonl').read_text()
            self.assertEqual(a,b);self.assertNotEqual(a,''.join(original))
            self.assertEqual(sorted(a.splitlines()),sorted(''.join(original).splitlines()))

    def test_substitution_is_same_class(self):
        self.assertEqual(apply_replacements('漢字©',[{'position':2,'from':'©','to':'〄'}]),'漢字〄')
        with self.assertRaises(ValueError):apply_replacements('漢字',[{'position':0,'from':'漢','to':'〄'}])


if __name__=='__main__':unittest.main()
