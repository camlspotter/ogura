import unittest

from ogura.textdet.prepare import merge_remaining_stacked_characters
from test_vertical_merge import fragment


def single(i, x=10, block=1):
    line = fragment(i, x, i * 10, wmode=0)
    line.update(baseline=(1., 0.), source_block=block)
    return line


class StackedCharactersTests(unittest.TestCase):
    def test_only_remaining_single_characters_are_combined(self):
        horizontal = single(4)
        horizontal['text'] = '横書き'
        horizontal['chars'] *= 3
        lines = [single(1), single(2), single(3), horizontal]
        out = merge_remaining_stacked_characters(lines)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]['text'], '123')
        self.assertEqual(out[0]['orientation'], 'vertical')
        self.assertEqual(out[0]['wmode'], 0)
        self.assertIs(out[1], horizontal)

    def test_different_columns_blocks_and_short_runs_stay_separate(self):
        for lines in ([single(1), single(2)],
                      [single(1), single(2, x=30), single(3)],
                      [single(1), single(2, block=2), single(3)]):
            self.assertEqual(merge_remaining_stacked_characters(lines), lines)

    def test_existing_line_interrupts_inference(self):
        horizontal = single(2)
        horizontal['text'] = '既存行'
        out = merge_remaining_stacked_characters([single(1), horizontal, single(3), single(4)])
        self.assertEqual(len(out), 4)
        self.assertIs(out[1], horizontal)
