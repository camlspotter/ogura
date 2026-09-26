import unittest
from ogura.textrec.build_evaluation_suite import choose, candidates
from ogura.textrec.training.aliases import CharacterAliases
from ogura.textrec.diversity import fingerprints
from ogura.textrec.training.train import selection_score


class EvaluationSuiteTests(unittest.TestCase):
    def test_selection_equal_sets_and_baseline_ignored(self):
        events=[dict(kind='validation_length',dataset=str(i),mode='augmented',cer=c)
                for i,c in enumerate([.01,.02,.03,.04,.05])]
        events += [dict(kind='validation_length',dataset='0',mode='baseline',cer=1)]
        self.assertAlmostEqual(selection_score('mean-set-cer',{'cer':0},events),.025)
        with self.assertRaises(ValueError):selection_score('mean-set-cer',{'cer':0},events+events[:1])
        with self.assertRaises(ValueError):selection_score('mean-set-cer',{'cer':0},[])

    def test_balancing_overlap_and_shortage(self):
        aliases=CharacterAliases({'version':1,'groups':[{'representative':'Ë','members':['Ë','Ё']}]})
        pool=[dict(text='abcdefghijklmnop'+c,language='en',article_id=str(i)) for i,c in enumerate('ËЁ')]
        pool += [dict(text='another different text ë',language='en',article_id='2')]
        forbidden=fingerprints('abcdefghijklmnopË');used=set()
        rows,counts=choose(pool,'ËЁë',3,forbidden,aliases,used,1)
        self.assertEqual([r['article_id'] for r in rows],['2'])
        self.assertEqual(counts,{'ë':1})
        self.assertEqual(choose(pool,'ËЁë',3,forbidden,aliases,used,1)[0],[])

    def test_source_offsets_and_table_rejection(self):
        text='解説\nこれはアルバニアの人物であり、名前はAbcëdefという表記で知られている人物である。'
        a=dict(id='1',title='人物',url='url',text=text)
        rows=list(candidates(a,set(text),'ë','ja'))
        self.assertTrue(rows)
        for r in rows:self.assertEqual(text[r['start']:r['end']],r['text'])
        a['title']='文字コード表'
        self.assertEqual(list(candidates(a,set(text),'ë','ja')),[])

    def test_build_writes_reserved_test_and_preserves_old_quotes(self):
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ogura.textrec.build_evaluation_suite import build
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'base';base.mkdir()
            (base/'train.txt').write_text('training-only\n')
            chars=set('“”‘’ËЁëёΠПΦФΓГ abcdefghijklmnopqrstuvwxyz')
            (base/'targets.jsonl').write_text(''.join(json.dumps({'character':c})+'\n' for c in sorted(chars)))
            alias=root/'aliases.json';alias.write_text(json.dumps({'version':1,'groups':[]}))
            for name in ('english_wikipedia_30k','quote_supplement'):
                p=root/'datasets'/name;p.mkdir(parents=True);(p/'train.jsonl').write_text('')
            q=base/'quotes';q.mkdir();old=dict(text='old “quote” with prose',article_id='old',title='old',url='url',start=0,end=21)
            (q/'validation.jsonl').write_text(json.dumps(old)+'\n');(q/'validation.txt').write_text(old['text']+'\n')
            jp=root/'corpus/wikipedia/20231101.ja';jp.mkdir(parents=True)
            articles=[dict(id=str(i),title='article',url='url',text='Ë“') for i in range(4)]
            pq.write_table(pa.Table.from_pylist(articles),jp/'a.parquet');en=root/'en.parquet'
            pq.write_table(pa.Table.from_pylist(articles[:0],schema=pa.Table.from_pylist(articles).schema),en)
            def excerpts(a,v,cs,lang):
                yield dict(text=chr(97+int(a['id']))*(25-len(cs))+cs,title='article',url='url',article_id=a['id'],start=0,end=25)
            args=SimpleNamespace(base=base,aliases=alias,output=root/'out',prior_probe=None,quote_goal=1,homoglyph_goal=1,test_goal=1,seed=1)
            with patch('ogura.textrec.build_evaluation_suite.ROOT',root),patch('ogura.textrec.build_evaluation_suite.CORPUS_ROOT',root/'corpus'),patch('ogura.textrec.build_evaluation_suite.download',return_value=(en,'url')),patch('ogura.textrec.build_evaluation_suite.split_article',side_effect=lambda aid,seed:'test' if int(aid)<2 else 'validation'),patch('ogura.textrec.build_evaluation_suite.candidates',side_effect=excerpts):
                build(args)
            vm=json.loads((args.output/'validation/quotes/manifest.json').read_text())
            tm=json.loads((args.output/'test/quotes/manifest.json').read_text())
            self.assertEqual(vm['samples'],2)
            self.assertEqual(tm['split'],'test')
            self.assertIn(old['text'],(args.output/'validation/quotes/validation.txt').read_text())
            self.assertTrue((args.output/'manifest.json').exists())
