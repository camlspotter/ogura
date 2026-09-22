import unittest
from ogura.training.metrics import weighted_edit_distance, edit_distance, batch_totals
from ogura.training.train import selection_score


class WeightedCERTests(unittest.TestCase):
    def test_quote_spaces_partial_credit(self):
        for quote in '“”‘’"\'':
            ref=f'a{quote}b'
            self.assertEqual(weighted_edit_distance(ref,f'a {quote} b'),1)
            self.assertEqual(edit_distance(ref,f'a {quote} b'),2)
            self.assertEqual(weighted_edit_distance(ref,ref),0)
        self.assertEqual(weighted_edit_distance('“x”',' “x” '),1)
        self.assertEqual(weighted_edit_distance('a“b','a  “b'),1)

    def test_other_errors_and_missing_quotes_full_cost(self):
        for ref,pred in [('ab','a b'),('a b','ab'),('a “b','a“b'),('ab','a“ b'),('',' '),('a','a ')]:
            self.assertEqual(weighted_edit_distance(ref,pred),edit_distance(ref,pred))
        self.assertEqual(weighted_edit_distance('a“b','a“x b'),2)
        self.assertEqual(batch_totals(['a “b'],['a“b'],0,1)['exact_matches'],0)

    def test_weighted_selection_opt_in(self):
        main={'cer':.1,'weighted_cer':.05}
        events=[dict(kind='validation_length',mode='augmented',dataset='quotes',cer=.3,weighted_cer=.15)]
        self.assertAlmostEqual(selection_score('mean-set-cer',main,events),.2)
        self.assertAlmostEqual(selection_score('mean-set-weighted-cer',main,events),.1)

    def test_evaluator_keeps_standard_metrics_and_ctc_loss(self):
        from unittest.mock import patch
        import torch
        from ogura.training.evaluate import evaluate
        from ogura.training.model import LineCNN
        from ogura.training.render import Vocabulary
        from ogura.training.train import TrainConfig, EpochDataset
        from tests.test_training import FONT
        if not FONT.exists():
            self.skipTest('Noto font required')
        torch.set_num_threads(1)
        vocabulary = Vocabulary('a“b ')
        dataset = EpochDataset([('a“b', 'quote')], [0], TrainConfig(font=FONT), epoch=0)
        model = LineCNN(len(vocabulary), channels=2)
        with patch('ogura.training.evaluate.decode', return_value=['a “b']):
            inserted = evaluate(model, dataset, vocabulary, 1, torch.device('cpu'))
        with patch('ogura.training.evaluate.decode', return_value=['a“b']):
            exact = evaluate(model, dataset, vocabulary, 1, torch.device('cpu'))
        self.assertEqual(inserted['character_errors'], 1)
        self.assertEqual(inserted['weighted_character_errors'], .5)
        self.assertEqual(inserted['weighted_cer'], inserted['cer']/2)
        self.assertEqual(inserted['exact_accuracy'], 0)
        self.assertEqual(exact['weighted_cer'], 0)
        self.assertEqual(exact['exact_accuracy'], 1)
        self.assertEqual(exact['mean_loss'], inserted['mean_loss'])

    def test_quote_substitutions_partial_credit(self):
        groups = ('“”"', "‘’'")
        for group in groups:
            for ref in group:
                for pred in group:
                    self.assertEqual(weighted_edit_distance(ref, pred), 0 if ref == pred else .5)
                    self.assertEqual(weighted_edit_distance('a'+ref+'b', 'a '+pred+' b'),
                                     1 if ref == pred else 1.5)
        for ref in groups[0]:
            for pred in groups[1]:
                self.assertEqual(weighted_edit_distance(ref, pred), 1)
                self.assertEqual(weighted_edit_distance(pred, ref), 1)
        self.assertEqual(weighted_edit_distance('“', ''), 1)
        self.assertEqual(weighted_edit_distance('', '“'), 1)
        totals = batch_totals(['a”b'], ['a“b'], 0, 1)
        self.assertEqual(totals['character_errors'], 1)
        self.assertEqual(totals['exact_matches'], 0)

    def test_two_single_quotes_and_one_double_quote(self):
        for a in "‘’'":
            for b in "‘’'":
                for double in '“”"':
                    pair = a+b
                    for ref, pred in [(pair, double), (double, pair)]:
                        self.assertEqual(weighted_edit_distance(ref, pred), .5)
                        self.assertEqual(weighted_edit_distance('前'+ref+'後', '前'+pred+'後'), .5)
                        self.assertEqual(weighted_edit_distance(ref, ref), 0)
                        self.assertEqual(edit_distance(ref, pred), 2)
                        self.assertEqual(weighted_edit_distance(ref+'x', pred+'y'), 1.5)

    def test_pairs_do_not_swallow_extra_quotes_or_internal_spaces(self):
        for count, expected in [(3, 1.5), (4, 2.5)]:
            singles = "'" * count
            self.assertEqual(weighted_edit_distance(singles, '"'), expected)
            self.assertEqual(weighted_edit_distance('"', singles), expected)
        self.assertEqual(weighted_edit_distance("''''", '""'), 1)
        self.assertEqual(weighted_edit_distance('""', "''''"), 1)
        self.assertEqual(weighted_edit_distance("' '", '"'), 3)
        self.assertEqual(weighted_edit_distance('"', "' '"), 3)
        self.assertEqual(weighted_edit_distance("''", "'"), 1)
        self.assertEqual(weighted_edit_distance("'", "''"), 1)
        self.assertEqual(weighted_edit_distance("a''“b", 'a" “b'), 1)
        self.assertEqual(weighted_edit_distance('a"“b', "a'' “b"), 1)
