import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

import torch

from ogura.training.metrics import MetricsLog, add_totals, batch_totals, decode, edit_distance, empty_totals, summary, epoch_eta, format_duration, local_finish_time
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

    def test_eta_uses_completed_epoch_totals_including_resumed_work(self):
        totals = empty_totals()
        self.assertIsNone(epoch_eta(totals, 10))
        totals.update(batches=4, seconds=8)
        self.assertEqual(epoch_eta(totals, 10), 12)
        self.assertEqual(epoch_eta(totals, 10, elapsed_seconds=20), 30)
        restored = json.loads(json.dumps(totals))
        add_totals(restored, batch_totals(['a'], ['a'], 1, 2))
        self.assertEqual(epoch_eta(restored, 10), 10)
        self.assertEqual(epoch_eta(restored, 5), 0)
        self.assertEqual(format_duration(3661.1), '01:01:02')
        self.assertEqual(format_duration(0), '00:00:00')
        self.assertEqual(format_duration(None), '--:--:--')
        finish = datetime.fromisoformat(local_finish_time(120, now=1700000000))
        self.assertIsNotNone(finish.utcoffset())
        self.assertEqual(finish.timestamp(), 1700000120)
        self.assertEqual(local_finish_time(None), '--')

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


class WorstSamplesTests(unittest.TestCase):
    def test_rank_by_rate_not_raw_errors_and_limit(self):
        from ogura.training.metrics import worst_samples
        references = ['日本語の文章です', '日', '本', '正解']
        predictions = ['日本語の文', '月火', '月', '正解']
        rows = worst_samples(references, predictions, 3)
        self.assertEqual([row[0] for row in rows], ['日', '本', '日本語の文章です'])
        self.assertEqual(rows[0][2], 2.0)
        self.assertEqual(worst_samples(references, predictions, 0), [])
        self.assertEqual(len(worst_samples(references, predictions, 99)), 4)
        self.assertEqual([r[0] for r in worst_samples(['日', '本'], ['月', '火'], 2)], ['日', '本'])
