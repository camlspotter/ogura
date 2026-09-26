import runpy
from pathlib import Path
import tempfile
import unittest

migrate = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/migrate_assets.py'))['migrate']


class AssetMigrationTests(unittest.TestCase):
    def test_move_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo/'textrec').mkdir(); (repo/'runs').mkdir()
            (repo/'runs/manifest.json').write_bytes(b'{"path":"runs/example"}')
            migrate(repo); migrate(repo)
            self.assertFalse((repo/'runs').exists())
            self.assertEqual((repo/'textrec/runs/manifest.json').read_bytes(), b'{"path":"runs/example"}')

    def test_collision_before_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo/'textrec/runs').mkdir(parents=True)
            (repo/'datasets').mkdir(); (repo/'runs').mkdir()
            with self.assertRaises(FileExistsError): migrate(repo)
            self.assertFalse((repo/'datasets').is_symlink())
            self.assertFalse((repo/'textrec/datasets').exists())
