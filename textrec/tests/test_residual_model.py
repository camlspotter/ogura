import unittest
import torch
from ogura.textrec.training.model import make_model, ResidualBlock, ctc_loss
from ogura.textrec.training.render import BatchRenderer, RenderParams, Sample, Vocabulary
from tests.test_training import FONT


class ResidualModelTests(unittest.TestCase):
    def test_proposed_architecture_parameter_count(self):
        model=make_model(16058,64,'residual')
        self.assertEqual(sum(p.numel() for p in model.parameters()),8235834)
        self.assertEqual(sum(isinstance(m,ResidualBlock) for m in model.modules()),6)
        self.assertEqual(sum(isinstance(m,torch.nn.Conv2d) for m in model.modules()),17)
        self.assertEqual(model.classifier.in_features,128)

    @unittest.skipUnless(FONT.exists(),'Noto font required')
    def test_short_normal_long_lines_and_finite_backward(self):
        torch.set_num_threads(1)
        vocabulary=Vocabulary('日本語文字')
        samples=[Sample(('日本語文字'*16)[:n],RenderParams(str(FONT),font_size=28)) for n in (5,25,80)]
        batch=BatchRenderer(vocabulary)(samples)
        model=make_model(len(vocabulary),64,'residual')
        output=model(batch.images)
        self.assertEqual(tuple(output.shape),((batch.images.shape[-1]+7)//8,3,len(vocabulary)))
        loss=ctc_loss(output,batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
