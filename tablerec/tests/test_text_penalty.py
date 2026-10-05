import unittest
import torch
from ogura.tablerec.table_cnn import text_false_positive_loss,collate_tables

class PenaltyTests(unittest.TestCase):
    def test_only_safe_text_receives_gradient(self):
        logits=torch.zeros(1,2,3,4,requires_grad=True)
        mask=torch.zeros(1,1,3,4);mask[:,:,1,1]=1;mask[:,:,2,3]=1
        valid=torch.ones_like(mask);valid[:,:,2,3]=0
        text_false_positive_loss(logits,mask,valid).backward()
        self.assertEqual(int((logits.grad!=0).sum()),2)
        self.assertTrue((logits.grad[:,:,1,1]>0).all())
        self.assertEqual(text_false_positive_loss(logits,mask*0,valid).item(),0)
    def test_padding_keeps_text_mask_zero(self):
        s=dict(image=torch.ones(3,3,4),target=torch.zeros(2,3,4),text_mask=torch.ones(1,3,4),id='a')
        b=collate_tables([s]);self.assertEqual(b['text_mask'].sum().item(),12)
        self.assertEqual(b['text_mask'][0,0,3:].sum().item(),0)
