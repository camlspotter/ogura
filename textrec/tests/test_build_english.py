import unittest
from ogura.textrec.build_english import candidate, prose, statistics


class EnglishExtractionTests(unittest.TestCase):
    def test_source_offsets_whole_words_and_repeatability(self):
        text = 'Heading\nThe office staff offered different flowers to the visitors and discussed the effects of the new official policy.'
        article = dict(id='1', title='Example', url='https://en.wikipedia.org/wiki/Example', text=text)
        row = candidate(article, set(text), 123)
        self.assertIsNotNone(row)
        self.assertEqual(row, candidate(article, set(text), 123))
        self.assertEqual(text[row['start']:row['end']], row['text'])
        self.assertTrue(20 <= len(row['text']) <= 25)
        self.assertTrue(row['start'] == 0 or text[row['start']-1].isspace())
        self.assertTrue(row['end'] == len(text) or text[row['end']].isspace())
        self.assertIsNone(candidate({**article, 'title':'List of flowers'}, set(text), 123))
        self.assertIsNone(candidate(article, set('abc'), 123))

    def test_tables_and_code_rejected(self):
        for text in ['A B C D E F G H I J K L M N O P.', 'x = { value | other }; '*20, 'https://example.com '+ 'the words '*20]:
            self.assertFalse(prose(text))

    def test_overlapping_counts(self):
        stats = statistics([dict(article_id='1', text='fff office')])
        self.assertEqual(stats['patterns']['ff']['occurrences'], 3)
        self.assertEqual(stats['patterns']['ffi']['occurrences'], 1)

    def test_build_is_repeatable_and_splits_articles(self):
        import argparse
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ogura.textrec.build_english import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            articles = [dict(id=str(i), title=f'Article {i}', url=f'https://example.com/{i}',
                             text=f'The visitors offered {i:05d} different flowers to the staff and discussed the effects of the new official policy.') for i in range(200)]
            corpus = root/'source.parquet'
            pq.write_table(pa.Table.from_pylist(articles), corpus)
            targets = root/'targets.jsonl'
            targets.write_text(''.join(json.dumps(dict(character=c))+'\n' for c in sorted(set(''.join(a['text'] for a in articles)))))
            old = root/'old.txt'; old.write_text('')
            def args(output):
                return argparse.Namespace(output=root/output, corpus=root, targets=targets, training_text=old,
                                          exclude_text=[], train_count=5, validation_count=2, seed=42)
            with patch('ogura.textrec.build_english.download', return_value=(corpus, 'test')):
                build(args('one')); build(args('two'))
            for name in ['train.txt','train.jsonl','validation.txt','validation.jsonl','manifest.json']:
                self.assertEqual((root/'one'/name).read_bytes(), (root/'two'/name).read_bytes())
            train = [json.loads(s) for s in (root/'one/train.jsonl').read_text().splitlines()]
            valid = [json.loads(s) for s in (root/'one/validation.jsonl').read_text().splitlines()]
            self.assertTrue({r['article_id'] for r in train}.isdisjoint(r['article_id'] for r in valid))
