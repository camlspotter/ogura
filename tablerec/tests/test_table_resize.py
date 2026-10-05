import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from ogura.tablerec.table_resize import capped_size,resize_image,resize_masks
from ogura.tablerec.synth_table_cells import generate
from ogura.tablerec.table_cnn import TableTrainingDataset
FONT=Path(__file__).resolve().parents[2]/'corpus/fonts/NotoSansCJKjp-Regular.otf'
class ResizeTests(unittest.TestCase):
    def test_cap_and_soft_mask(self):
        self.assertEqual(capped_size((2000,1000),1024),(1024,512))
        self.assertEqual(capped_size((512,2048),1024),(256,1024))
        self.assertEqual(capped_size((300,200),1024),(300,200))
        for value in [0,-1,1.5]:
            with self.assertRaises(ValueError):capped_size((1,1),value)
        m=np.zeros((2,8,8),dtype=np.float32);m[0,:,3]=1
        reduced=resize_masks(m,(4,4))
        self.assertAlmostEqual(float(reduced[0].sum()),2)
        self.assertTrue(np.any((reduced>0)&(reduced<1)))
    @unittest.skipUnless(FONT.exists(),'Japanese font')
    def test_training_geometry_and_text_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';generate(root,1,44,'train',[FONT],background_context=True)
            before=(root/'recipes/table-000000.json').read_bytes()
            data=TableTrainingDataset(root,max_side=128,text_penalty=True)
            sample=data[0];self.assertLessEqual(max(sample['image'].shape[-2:]),128)
            self.assertEqual(tuple(sample['image'].shape[-2:]),data.sizes[0])
            self.assertEqual(sample['target'].shape[-2:],sample['text_mask'].shape[-2:])
            self.assertFalse(((sample['target'].max(0).values>0)&(sample['text_mask'][0]>0)).any())
            self.assertEqual((root/'recipes/table-000000.json').read_bytes(),before)
