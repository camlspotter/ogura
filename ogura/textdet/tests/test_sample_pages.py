import unittest

from ogura.textdet.sample_pages import choose_pages


class SamplePagesTests(unittest.TestCase):
    def test_only_last_page_has_text(self):
        self.assertEqual(choose_pages([0] * 12 + [1589]), [13])

    def test_quantiles_use_only_nonempty_pages(self):
        self.assertEqual(choose_pages([0, 10, 0, 20, 0, 30, 0, 40, 0, 50, 0, 60]), [4, 8])

    def test_all_pages_have_text(self):
        self.assertEqual(choose_pages([10] * 13), [5, 9])
        self.assertEqual(choose_pages([10, 20]), [1, 2])

    def test_no_text(self):
        self.assertEqual(choose_pages([0, 0]), [])


if __name__ == '__main__':
    unittest.main()
