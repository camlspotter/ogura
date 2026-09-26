import unittest
from ogura.training.metrics import evaluation_texts, batch_totals, weighted_edit_distance
from ogura.training.scoring import allow_punctuation_space


class PeriodSpaceTests(unittest.TestCase):
    def test_allowed_insertion(self):
        for ref, pred in [('1.感度', '1. 感度'), ('a.b.c', 'a. b. c'), ('.', '. ')]:
            self.assertEqual(allow_punctuation_space(ref, pred), ref)
            self.assertEqual(batch_totals([pred], [ref], 0, 1)['exact_matches'], 1)

    def test_other_errors_preserved(self):
        for ref, pred in [('a. b', 'a.b'), ('a,b', 'a, b'), ('ab', 'a b'), ('1.感度', '1. 感都')]:
            normalized = allow_punctuation_space(ref, pred)
            self.assertNotEqual(normalized, ref)
        self.assertEqual(allow_punctuation_space('a. b', 'a. b'), 'a. b')

    def test_parenthesis_spaces(self):
        for ref, pred in [(')(', ') ('), ('）文字', '） 文字'),
                          ('字（文', '字 （文'), ('(文', ' (文'),
                          ('）', '） '), (')(文)', ') (文)')]:
            self.assertEqual(allow_punctuation_space(ref, pred), ref)
            self.assertEqual(batch_totals([pred], [ref], 0, 1)['exact_matches'], 1)

    def test_parenthesis_does_not_hide_other_spaces(self):
        for ref, pred in [('(文)', '( 文)'), ('(文)', '(文 )'),
                          ('通知して', '通知 して'), ('1105001', '1 105001'),
                          (') (', ')(')]:
            self.assertNotEqual(allow_punctuation_space(ref, pred), ref)

    def test_scoring_copies_only(self):
        refs, preds = ['1.感度'], ['1. 感度']
        r, p = evaluation_texts(refs, preds)
        self.assertEqual(weighted_edit_distance(r[0], p[0]), 0)
        self.assertEqual(preds, ['1. 感度'])

if __name__ == '__main__': unittest.main()
