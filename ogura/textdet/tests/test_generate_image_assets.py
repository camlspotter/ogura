import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import tempfile
from PIL import Image
from ogura.textdet.generate_image_assets import ROOT, read_prompts, generate


class ImageAssetTests(TestCase):
    def test_sample_prompts_and_duplicate_rejection(self):
        self.assertEqual(len(read_prompts(ROOT/'prompts/image_assets_sample.jsonl')), 10)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'prompts.jsonl'
            p.write_text('{"id":"a","prompt":"one"}\n{"id":"a","prompt":"two"}\n')
            with self.assertRaises(ValueError):
                read_prompts(p)

    def test_generation_arguments_resume_and_corruption_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(size=64, seed=42, output=Path(tmp)/'out',
                                   cache=Path(tmp)/'cache', resume=False, cpu_offload=False)
            rows = [dict(id='one', prompt='a plain leaf'), dict(id='two', prompt='a plain stone')]
            pipe = Mock(return_value=SimpleNamespace(images=[Image.new('RGB',(64,64),'green')]))
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.is_bf16_supported', return_value=True), \
                 patch('torch.cuda.init') as cuda_init, \
                 patch('torch.empty') as cuda_probe, \
                 patch('torch.cuda.synchronize'), \
                 patch('torch.cuda.mem_get_info', return_value=(48 * 2**30, 119 * 2**30)), \
                 patch('torch.Generator'), \
                 patch('ogura.textdet.generate_image_assets.download', return_value='fake') as download, \
                 patch('diffusers.ZImagePipeline.from_pretrained', return_value=pipe) as load:
                def check_context_before_load(*a, **kw):
                    cuda_init.assert_called_once_with()
                    cuda_probe.assert_called_once_with(1, device='cuda')
                    return pipe
                load.side_effect = check_context_before_load
                generate(args, rows)
                self.assertEqual(pipe.call_count, 2)
                self.assertEqual(pipe.call_args.kwargs['num_inference_steps'], 9)
                self.assertEqual(pipe.call_args.kwargs['guidance_scale'], 0)
                self.assertTrue((args.output/'contact-sheet.jpg').exists())
                self.assertEqual(json.loads((args.output/'manifest.json').read_text())['status'], 'complete')
                with self.assertRaises(FileExistsError):
                    generate(args, rows)
                args.resume = True
                generate(args, rows)
                self.assertEqual(pipe.call_count, 2)
                self.assertEqual(download.call_count, 1)
                self.assertEqual(cuda_init.call_count, 1)
                args.seed = 43
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    generate(args, rows)
                args.seed = 42
                (args.output/'images/one.png').write_bytes(b'broken')
                with self.assertRaisesRegex(ValueError, 'Image changed'):
                    generate(args, rows)
