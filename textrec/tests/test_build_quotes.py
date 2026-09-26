import unittest
from ogura.textrec.build_english import candidate
from ogura.textrec.build_quotes import select


class QuoteExtractionTests(unittest.TestCase):
    def test_literal_targeted_window(self):
        text='The author wrote “a beautiful day” in her book and described the flowers as ‘a lovely surprise’ to the visitors.'
        article=dict(id='1',title='Example',url='example',text=text)
        for c in '“”‘’':
            row=candidate(article,set(text),42,required_character=c)
            self.assertIsNotNone(row)
            self.assertIn(c,row['text'])
            self.assertEqual(text[row['start']:row['end']],row['text'])
            self.assertTrue(20<=len(row['text'])<=25)

    def test_balanced_distinct_selection_and_shortfall(self):
        pool=[dict(article_id=str(i),text=c+str(i)*24) for i,c in enumerate('“”‘’')]
        selected=select(pool[:],1,set(),42)
        self.assertEqual(len(selected),4)
        self.assertEqual(len({r['article_id'] for r in selected}),4)
        with self.assertRaises(ValueError):select(pool[:],2,set(),42)
