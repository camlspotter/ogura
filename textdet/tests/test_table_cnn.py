import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image

from ogura.textdet.table_cnn import (CHANNELS, SizeBatchSampler, TableUNet, boundary_loss,
    boundary_metrics, collate_tables, load_model, predict_image)


class TableCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_shapes_backward_and_checkpoint(self):
        model = TableUNet(4)
        x = torch.rand(2, 3, 37, 51)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 2, 37, 51))
        target = torch.rand_like(logits)
        loss = boundary_loss(logits, target, torch.ones(2, 1, 37, 51))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            torch.save(dict(format_version=1, channels=list(CHANNELS), base_channels=4, model=model.state_dict()), path)
            restored = load_model(path)
            self.assertTrue(torch.equal(model(x), restored(x)))
            result = predict_image(restored, Image.new('RGB', (51, 37), 'white'))
            self.assertEqual(result.shape, (2, 37, 51))
            self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_padding_is_excluded_from_loss_metrics_and_gradients(self):
        logits = torch.randn(2, 2, 17, 21)
        target = torch.rand_like(logits)
        mask = torch.ones(2, 1, 17, 21)
        expected = boundary_loss(logits, target, mask)
        padded = F.pad(logits, (0, 11, 0, 15), value=12).requires_grad_()
        padded_target = F.pad(target, (0, 11, 0, 15), value=0.7)
        padded_mask = F.pad(mask, (0, 11, 0, 15))
        actual = boundary_loss(padded, padded_target, padded_mask)
        self.assertTrue(torch.allclose(expected, actual))
        a = boundary_metrics(logits, target, mask)
        b = boundary_metrics(padded.detach(), padded_target, padded_mask)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=6)
        actual.backward()
        self.assertEqual(padded.grad[..., 17:, :].abs().sum().item(), 0)
        self.assertEqual(padded.grad[..., :, 21:].abs().sum().item(), 0)

    def test_batch_shapes_and_sampler_coverage(self):
        samples = [dict(image=torch.zeros(3,h,w), target=torch.full((2,h,w),0.25), id=str(i))
                   for i,(h,w) in enumerate([(17,21),(33,40)])]
        batch = collate_tables(samples)
        self.assertEqual(batch['image'].shape, (2,3,48,48))
        self.assertEqual(batch['valid'].sum().item(), 17*21+33*40)
        self.assertEqual(batch['target'][0,0,0,0].item(), 0.25)
        self.assertEqual(batch['image'][0,:,40,40].tolist(), [1,1,1])
        sampler = SizeBatchSampler([(20+i, 30+i) for i in range(71)], 3)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        self.assertEqual(sorted(i for b in first for i in b), list(range(71)))
        sampler.epoch = 1
        self.assertNotEqual(first, list(sampler))
        self.assertEqual(len(first), len(sampler))

    def test_tiny_ruled_image_can_be_learned(self):
        torch.manual_seed(7)
        target = torch.zeros(1,2,32,32)
        target[:,0,8,:] = 1
        target[:,0,24,:] = 0.5
        target[:,1,:,16] = 1
        x = (1-target.amax(1, keepdim=True)).repeat(1,3,1,1)
        valid = torch.ones(1,1,32,32)
        model = TableUNet(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        start = boundary_loss(model(x), target, valid).item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = boundary_loss(model(x), target, valid)
            loss.backward()
            optimizer.step()
        end = boundary_loss(model(x), target, valid).item()
        self.assertLess(end, start*0.7)


if __name__ == '__main__':
    unittest.main()
