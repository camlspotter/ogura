import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ogura.textrec.download_training_fonts import entries, fetch, font_paths


class DownloadTests(unittest.TestCase):
    def test_pinned_download_cache_and_reject_changes(self):
        data = b'font fixture'
        entry = dict(name='font.ttf', url='https://example.invalid/font.ttf',
                     sha256=hashlib.sha256(data).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(data)):
                target = fetch(entry, output)
            self.assertEqual(target.read_bytes(), data)
            with patch('urllib.request.urlopen', side_effect=AssertionError('No network needed')):
                self.assertEqual(fetch(entry, output), target)
                target.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    fetch(entry, output)
            target.unlink()
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    fetch(entry, output)
            self.assertFalse(target.exists())
            self.assertEqual(list(output.iterdir()), [])

    def test_selected_fonts_and_licenses(self):
        self.assertEqual(len(font_paths(Path('.'))), 4)
        self.assertEqual(len(font_paths(Path('.'), True)), 7)
        selected = entries(True)
        self.assertIn('LICENSE-ZenMaruGothic', [e['name'] for e in selected])
        self.assertNotIn('MPLUSRounded1c-Light.ttf', [e['name'] for e in selected])
        extra = entries(True, True, True)
        names = {e['name'] for e in extra}
        self.assertTrue({'MPLUSRounded1c-Thin.ttf', 'MPLUSRounded1c-Light.ttf', 'KosugiMaru-Regular.ttf', 'LICENSE-KosugiMaru.txt'} <= names)
        self.assertEqual(len(font_paths(Path('.'), True, True)), 10)
        self.assertEqual(len(names), len(extra))
        for entry in extra:
            self.assertEqual(len(entry['sha256']), 64)
            self.assertNotIn('/main/', entry['url'])
