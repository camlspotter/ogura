"""Public entry points and resources survive the recognition package relocation."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest


class TextrecLayoutTests(unittest.TestCase):
    def test_legacy_modules_and_resources(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('text_common', 'analyze_wikipedia', 'classify_characters',
                     'verify_training_text', 'refine_charset', 'evaluate_real'):
            for prefix in ('ogura.', 'ogura.textrec.'):
                self.assertEqual(importlib.import_module(prefix + name).ROOT, root)
        for prefix in ('ogura.', 'ogura.textrec.'):
            catalog = importlib.import_module(prefix + 'download_training_fonts').CATALOG
            self.assertTrue(catalog.is_file())
            self.assertEqual(catalog.parent, root / 'ogura/textrec')

    def test_old_and_new_cli(self):
        for name in ('training.train', 'recognize'):
            for prefix in ('ogura.', 'ogura.textrec.'):
                result = subprocess.run([sys.executable, '-m', prefix + name, '--help'],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
