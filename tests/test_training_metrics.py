import json
from pathlib import Path
import tempfile
import unittest

import torch

from ogura.training.metrics import MetricsLog, add_totals, batch_totals, decode, edit_distance, empty_totals, summary
from ogura.training.render import Vocabulary


class MetricsTests(unittest.TestCase):
    def test_ctc_decode_collapses_before_removing_blank_and_ignores_padding(self):
        paths = torch.tensor([[1,1,0,1,2,2,0,2], [0,2,2,0,1,1,2,2]])
        logits = torch.nn.functional.one_hot(paths.T, 3).float()
        self.assertEqual(decode(logits, torch.tensor([8,4]), Vocabulary('ab')), ['aabb','b'])

    def test_weighted_accuracy_and_character_error_rate(self):
        totals = empty_totals()
        add_totals(totals, batch_totals(['abc'], ['abc'], 2, 1))
        add_totals(totals, batch_totals(['axc','aaa',''], ['abc','a','bb'], 4, 3))
        result = summary(totals)
        self.assertEqual(result['exact_accuracy'], 0.25)
        self.assertEqual(result['cer'], 5/9)
        self.assertEqual(result['mean_loss'], 3.5)
        self.assertEqual(result['seconds'], 4)
        self.assertEqual(result['samples_per_second'], 1)
        self.assertEqual(edit_distance('a','aaaa'),3)  # CER may exceed 100%.

    def test_log_rolls_back_partial_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metrics.jsonl'
            log=MetricsLog(path)
            log.append([{'step':1}]); saved=log.offset
            log.append([{'step':2}])
            with path.open('ab') as stream:stream.write(b'{incomplete')
            resumed=MetricsLog(path,saved)
            resumed.append([{'step':2}])
            self.assertEqual([json.loads(line)['step'] for line in path.read_text().splitlines()],[1,2])
            with self.assertRaises(ValueError):MetricsLog(path,path.stat().st_size+1)
