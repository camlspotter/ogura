import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from ogura.textrec.collect_latin_sequences import build, excerpts, matched_words, training_article


class LatinSequenceCollectionTests(unittest.TestCase):
    def test_literal_whole_token_excerpts_and_patterns(self):
        text='The office staff offered different flowers to the visitors and discussed the effects of the new official policy.'
        rows=list(excerpts(text,7,set(text)))
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(text[row['start']-7:row['end']-7],row['text'])
            self.assertTrue(20<=len(row['text'])<=25)
            self.assertTrue(row['patterns'])
            start,end=row['start']-7,row['end']-7
            self.assertTrue(start==0 or text[start-1].isspace())
            self.assertTrue(end==len(text) or text[end].isspace())
        self.assertEqual(list(excerpts(text,0,set('abc'))),[])
        self.assertEqual([m.group() for m in matched_words('billié Xff2 ff_aa office')], ['office'])
        self.assertEqual([m.group() for m in matched_words('II III Hawaii Fiji Jilin ji')], ['Hawaii', 'Fiji', 'Jilin', 'ji'])
        self.assertTrue(any({'ff','fi','ffi'}<=set(r['patterns']) for r in rows))

    def test_build_repeatable_source_provenance_and_heldout_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            articles=[dict(id=str(i),title=f'Article {i}',url=f'https://en.wikipedia.org/wiki/{i}',
                           text=f'The office staff offered {i:05d} different flowers to the visitors and discussed the effects of the official policy.') for i in range(80)]
            source=root/'input.parquet';pq.write_table(pa.Table.from_pylist(articles),source)
            targets=root/'targets.jsonl';targets.write_text(''.join(json.dumps({'character':c})+'\n' for c in sorted(set(''.join(a['text'] for a in articles)))))
            aliases=root/'aliases.json';aliases.write_text('{"version": 1, "groups": []}')
            train=root/'train.txt';train.write_text('')
            held=next(a for a in articles if training_article(a['id']))
            exclude=root/'validation.jsonl';exclude.write_text(json.dumps({**held,'article_id':held['id'],'language':'en','text':'A held-out excerpt.'})+'\n')
            def args(name):return argparse.Namespace(corpus=root,output=root/name,goal=2,aliases=aliases,targets=targets,training_text=train,exclude_jsonl=[exclude])
            with patch('ogura.textrec.collect_latin_sequences.download',return_value=(source,'fixture')):
                build(args('first'));build(args('second'))
            for name in ('train.txt','train.jsonl','words.jsonl','manifest.json'):
                self.assertEqual((root/'first'/name).read_bytes(),(root/'second'/name).read_bytes())
            rows=[json.loads(s) for s in (root/'first/train.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            for row in rows:
                self.assertTrue(training_article(row['article_id']))
                self.assertNotEqual(row['article_id'],held['id'])
                original=articles[int(row['article_id'])]['text']
                self.assertEqual(original[row['start']:row['end']],row['text'])
            self.assertEqual(len(rows),len({r['text'] for r in rows}))
            with self.assertRaises(FileExistsError):build(args('first'))
