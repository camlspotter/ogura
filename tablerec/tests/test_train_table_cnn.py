import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ogura.tablerec.table_cnn import CHANNELS, TableUNet
from ogura.tablerec.train_table_cnn import initialize_model, run_epoch, train


class FineTuneTests(unittest.TestCase):
    def test_initial_weights_and_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'initial.pt'
            original = TableUNet(4)
            torch.save(dict(format_version=1, channels=list(CHANNELS),
                            base_channels=4, model=original.state_dict()), checkpoint)
            loaded = initialize_model(None, checkpoint, 'cpu')
            self.assertEqual(loaded.base_channels, 4)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            with self.assertRaisesRegex(ValueError, 'base-channels'):
                initialize_model(8, checkpoint, 'cpu')
            self.assertEqual(initialize_model(4, None, 'cpu').base_channels, 4)

    def test_fine_tuning_starts_fresh_and_saves_loadable_checkpoint(self):
        class TinyDataset:
            def __init__(self, path, **kwargs):
                self.seeds = {str(path)}
                self.sizes = [(32, 32)]
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return dict(image=torch.ones(3, 32, 32), target=torch.zeros(2, 32, 32), id='tiny')

        old_threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                initial = root/'initial.pt'
                original = TableUNet(4).state_dict()
                torch.save(dict(format_version=1, channels=list(CHANNELS), base_channels=4,
                                model=original, optimizer={'not': 'an optimizer'}, epoch=50), initial)
                before = initial.read_bytes()
                args = argparse.Namespace(train=root/'train', validation=root/'validation',
                    output=root/'output', init_checkpoint=initial, base_channels=None,
                    font_dir=None, device='cpu', epochs=1, batch_size=1, lr=1e-4,
                    workers=0, threads=1, seed=42, train_limit=None, validation_limit=None,
                    max_train_batches=None, max_validation_batches=None)
                def checked_epoch(model, loader, device, **kwargs):
                    optimizer = kwargs.get('optimizer')
                    if optimizer is not None:
                        self.assertEqual(len(optimizer.state), 0)
                        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
                        for key, value in original.items():
                            self.assertTrue(torch.equal(value, model.state_dict()[key]))
                    return run_epoch(model, loader, device, **kwargs)
                with patch('ogura.tablerec.train_table_cnn.TableTrainingDataset', TinyDataset), \
                     patch('ogura.tablerec.train_table_cnn.dataset_digest', return_value='tiny'), \
                     patch('ogura.tablerec.train_table_cnn.run_epoch', side_effect=checked_epoch):
                    train(args)
                saved = torch.load(args.output/'best.pt', weights_only=True)
                self.assertEqual(saved['epoch'], 1)
                self.assertEqual(saved['base_channels'], 4)
                self.assertEqual(saved['config']['init_checkpoint_sha256'], hashlib.sha256(before).hexdigest())
                self.assertTrue(any(not torch.equal(v, saved['model'][k]) for k, v in original.items()))
                self.assertEqual(initial.read_bytes(), before)
                self.assertEqual(initialize_model(None, args.output/'best.pt', 'cpu').base_channels, 4)
        finally:
            torch.set_num_threads(old_threads)
