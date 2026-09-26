"""CPU integration tests for rendering, actual CTC updates and crash recovery."""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

from ogura.textrec.text_common import ROOT
from ogura.textrec.training.checkpoint import Checkpoints, IdentityMismatch
from ogura.textrec.training.model import LineCNN, ctc_loss
from ogura.textrec.training.render import BatchRenderer, RenderParams, Sample, Vocabulary, parameters_for_sample, render_sample
from ogura.textrec.training.train import TrainConfig, train, prepare_data, load_initial_weights

FONT = ROOT.parent / 'corpus/fonts/NotoSansCJKjp-Regular.otf'


def minimal_state(step):
    return dict(identity={'test': 1}, model={}, optimizer={}, scheduler={}, scaler={},
                position={'step': step}, rng={})


class CheckpointTests(unittest.TestCase):
    def test_atomic_save_failure_and_corruption_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = Checkpoints(tmp)
            manager.save(minimal_state(1))
            manager.save(minimal_state(2))
            def fail(state, stream):
                stream.write(b'incomplete')
                raise OSError('simulated disk failure')
            with patch('ogura.textrec.training.checkpoint.torch.save', side_effect=fail):
                with self.assertRaises(OSError):
                    manager.save(minimal_state(3))
            self.assertEqual(manager.load({'test': 1})['position']['step'], 2)
            manager.latest.write_bytes(b'corrupt checkpoint')
            with warnings.catch_warnings(record=True) as caught:
                self.assertEqual(manager.load({'test': 1})['position']['step'], 1)
                self.assertTrue(caught)
            # A recovered previous checkpoint must survive the next save.
            manager.save(minimal_state(4))
            self.assertEqual(torch.load(manager.previous, weights_only=True)['position']['step'], 1)
            with self.assertRaises(IdentityMismatch):
                manager.load({'test': 2})
            self.assertFalse(list(Path(tmp).glob('*.tmp')))

    def test_only_known_logging_revision_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = Checkpoints(tmp)
            state = minimal_state(1)
            state['identity']['training_code_sha256'] = 'old'
            manager.save(state)
            current = {'test': 1, 'training_code_sha256': 'new'}
            with self.assertRaises(IdentityMismatch):manager.load(current)
            self.assertEqual(manager.load(current, ('old',))['position']['step'], 1)
            with self.assertRaises(IdentityMismatch):manager.load({**current, 'test': 2}, ('old',))

    def test_interruption_between_checkpoint_renames(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = Checkpoints(tmp)
            manager.save(minimal_state(1))
            import os
            real_replace = os.replace
            def fail_new(source, destination):
                if Path(source).name.startswith('.checkpoint-'):
                    raise OSError('power loss after rotation')
                return real_replace(source, destination)
            with patch('ogura.textrec.training.checkpoint.os.replace', side_effect=fail_new):
                with self.assertRaises(OSError):manager.save(minimal_state(2))
            with warnings.catch_warnings(record=True):
                state = Checkpoints(tmp).load({'test': 1})
            self.assertEqual(state['position']['step'], 1)


@unittest.skipUnless(FONT.exists(), 'Copy NotoSansCJKjp-Regular.otf into corpus/fonts first')
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_render_batch_labels_padding_and_missing_glyph(self):
        vocabulary = Vocabulary('日本語の文字')
        render = BatchRenderer(vocabulary)
        samples = [Sample('日本語', RenderParams(str(FONT)), 'a'),
                   Sample('文字', RenderParams(str(FONT)), 'b')]
        batch = render(samples)
        self.assertEqual(batch.images.shape[:3], (2, 1, 48))
        self.assertEqual(batch.target_lengths.tolist(), [3, 2])
        self.assertEqual(batch.targets.tolist(), vocabulary.encode('日本語文字'))
        self.assertGreater(batch.image_widths[0], batch.image_widths[1])
        self.assertTrue(torch.all(batch.images[1, :, :, batch.image_widths[1]:] == 1))
        self.assertLess(float(batch.images.min()), 1)
        self.assertTrue(torch.equal(batch.images, render(samples).images))
        with self.assertRaisesRegex(ValueError, 'No renderable text'):
            render_sample(Sample('\U0010ffff', RenderParams(str(FONT))))
        with self.assertRaises(KeyError):render([Sample('外', RenderParams(str(FONT)))])
        p = parameters_for_sample(FONT, 7, 2, 'a', 32, 40)
        random.seed(99); random.random()
        self.assertEqual(p, parameters_for_sample(FONT, 7, 2, 'a', 32, 40))

    def test_missing_glyph_replaces_image_and_label_without_dropping_rows(self):
        missing = '\U0010ffff'
        original = missing + '日' + missing * 2 + '本' + missing
        expected = '日 本'
        vocabulary = Vocabulary(' 日本' + missing)
        renderer = BatchRenderer(vocabulary)
        batch = renderer([Sample(original, RenderParams(str(FONT)), 'original-id')])
        reference = renderer([Sample(expected, RenderParams(str(FONT)))])
        self.assertEqual(batch.texts, [expected])
        self.assertEqual(batch.sample_ids, ['original-id'])
        self.assertEqual(batch.targets.tolist(), vocabulary.encode(expected))
        self.assertEqual(batch.target_lengths.tolist(), [len(expected)])
        self.assertTrue(torch.equal(batch.images, reference.images))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'text.txt'
            path.write_text(original + '\n日本\n')
            records, report = prepare_data(TrainConfig(text=path, font=FONT), vocabulary)
            self.assertEqual([r[0] for r in records], [expected, '日本'])
            self.assertEqual(report['training_rows'], 2)
            self.assertEqual(report['excluded_rows'], 0)
            self.assertEqual(report['replaced_rows'], 1)
            self.assertEqual(report['replaced_characters'], 4)
            self.assertEqual(report['replaced_line_numbers'], [1])
            with self.assertRaisesRegex(ValueError, 'space'):
                prepare_data(TrainConfig(text=path, font=FONT), Vocabulary('日本' + missing))

    def test_augmentation_replay_and_font_specific_labels(self):
        params = [parameters_for_sample(FONT, 7, 2, str(i), 32, 40,
                  ('second-font',), 2, 6, 3, 0.25) for i in range(100)]
        self.assertEqual(params, [parameters_for_sample(FONT, 7, 2, str(i), 32, 40,
                         ('second-font',), 2, 6, 3, 0.25) for i in range(100)])
        self.assertGreater(len(set(params)), 10)
        self.assertIn(RenderParams(str(FONT)), params)
        self.assertEqual({p.font_path for p in params}, {str(FONT), 'second-font'})
        self.assertNotEqual(params[0:10], [parameters_for_sample(FONT, 7, 3, str(i), 32, 40,
                            ('second-font',), 2, 6, 3, 0.25) for i in range(10)])
        for p in params:
            if p.font_path != str(FONT): continue
            im = render_sample(Sample('日本語', p))
            from PIL import ImageChops
            box = ImageChops.invert(im).getbbox()
            self.assertGreaterEqual(box[1], p.padding)
            self.assertLessEqual(box[3], 48-p.padding)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'text.txt'
            path.write_text('日本\n')
            config = TrainConfig(text=path, font=FONT, extra_fonts=(Path('second-font'),))
            def coverage(font):
                return frozenset(map(ord, ' 日本' if font == str(FONT) else ' 日'))
            with patch('ogura.textrec.training.train.font_characters', side_effect=coverage):
                records, _ = prepare_data(config, Vocabulary(' 日本'))
            self.assertEqual(records[0][0], '日本')
            with patch('ogura.textrec.training.render.font_characters', side_effect=coverage), \
                 patch('ogura.textrec.training.render.load_font', return_value=__import__('ogura.textrec.training.render', fromlist=['load_font']).load_font(str(FONT),40)):
                batch = BatchRenderer(Vocabulary(' 日本'))([
                    Sample(records[0][0], RenderParams(str(FONT))),
                    Sample(records[0][0], RenderParams('second-font'))])
            self.assertEqual(batch.texts, ['日本', '日'])

    def test_warm_start_checks_label_order_and_loads_exact_weights(self):
        vocab = Vocabulary(' 日本')
        original = LineCNN(len(vocab), channels=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'best.pt'
            torch.save(dict(characters=list(vocab.characters), channels=2, model=original.state_dict()), path)
            target = LineCNN(len(vocab), channels=2)
            load_initial_weights(target, path, vocab, 2)
            self.assert_nested_equal(original.state_dict(), target.state_dict())
            with self.assertRaises(ValueError):
                load_initial_weights(target, path, Vocabulary(' 本日'), 2)

    def test_independent_horizontal_padding_preserves_pixels(self):
        p = RenderParams(str(FONT), padding=4, padding_left=2, padding_right=6)
        a = render_sample(Sample('日本語', p))
        b = render_sample(Sample('日本語', replace(p, padding_left=6, padding_right=2)))
        self.assertEqual(a.size, b.size)
        self.assertEqual(a.crop((2,0,a.width-6,48)).tobytes(),
                         b.crop((6,0,b.width-2,48)).tobytes())
        self.assertEqual(a.crop((0,0,2,48)).getextrema(), (255,255))
        self.assertEqual(b.crop((b.width-2,0,b.width,48)).getextrema(), (255,255))
        with self.assertRaises(ValueError):
            render_sample(Sample('日本語', replace(p, padding_left=-1)))

    def test_full_vertical_range_preserves_ink_and_reaches_both_edges(self):
        from PIL import ImageChops
        p = RenderParams(str(FONT), font_size=28, padding=2, vertical_position=0)
        top = render_sample(Sample('日本語', p))
        bottom = render_sample(Sample('日本語', replace(p, vertical_position=1)))
        a = ImageChops.invert(top).getbbox()
        b = ImageChops.invert(bottom).getbbox()
        self.assertEqual(a[1], 2)
        self.assertEqual(b[3], 46)
        self.assertGreater(b[1]-a[1], 6)
        self.assertEqual(top.crop(a).tobytes(), bottom.crop(b).tobytes())

    def test_ctc_lengths_and_finite_gradients(self):
        batch = BatchRenderer(Vocabulary('日本'))([Sample('日日本', RenderParams(str(FONT)))])
        model = LineCNN(3, channels=2)
        logits = model(batch.images)
        self.assertEqual(logits.shape[0], (batch.images.shape[-1] + 7) // 8)
        loss = ctc_loss(logits, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        batch.image_widths[:] = 8
        with self.assertRaises(ValueError):ctc_loss(logits, batch)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x,y in zip(a,b):self.assert_nested_equal(x,y)
        else:self.assertEqual(a,b)

    def test_crash_replay_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            texts = ['日本語', '明日は雨', '今日は晴', '東京の空', '文字認識', '科学研究', '大阪の町', '山と川']
            (root/'text.txt').write_text('\n'.join(texts)+'\n')
            (root/'targets.jsonl').write_text(''.join(json.dumps({'character':c},ensure_ascii=False)+'\n' for c in sorted(set(''.join(texts)))))
            config = TrainConfig(text=root/'text.txt', vocabulary=root/'targets.jsonl', font=FONT,
                                 run_dir=root/'full', device='cpu', batch_size=2, epochs=2,
                                 save_every=2, channels=2, model_type='residual', threads=1, log_every=99,
                                 deterministic=True, font_size_min=32, font_size_max=40, log_samples=1,
                                 extra_fonts=(FONT,), padding_min=2, padding_max=6, vertical_jitter=3, clean_probability=0.25, vertical_full_range=True)
            full_ids = []
            output = io.StringIO()
            with redirect_stdout(output):
                train(config, lambda pos,ids: full_ids.append(ids))
            self.assertEqual(output.getvalue().count('  正解:'), 1)
            self.assertEqual(output.getvalue().count('  予測:'), 1)
            broken = replace(config, run_dir=root/'broken')
            def crash(pos, ids):
                if pos['step'] == 3:raise RuntimeError('simulated crash after unsaved update')
            with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                train(broken, crash)
            saved = torch.load(broken.run_dir/'latest.pt', weights_only=True)
            self.assertEqual(saved['position'], {'step':2,'epoch':0,'next_batch':2})
            self.assertEqual(saved['metrics']['epoch_totals']['samples'], 4)
            self.assertGreater(saved['metrics']['epoch_elapsed_seconds'], 0)
            with (broken.run_dir/'metrics.jsonl').open('ab') as stream:
                stream.write(b'{partial event after crash')
            resumed_ids = []
            # Prefetch workers differ from the original run; batch position still replays exactly.
            output = io.StringIO()
            with redirect_stdout(output):
                train(replace(broken, resume=True, workers=2, log_samples=0, log_every=1), lambda pos,ids: resumed_ids.append(ids))
            self.assertNotIn('  正解:', output.getvalue())
            self.assertIn('step=3', output.getvalue())
            self.assertIn('batch=3/4 (75.0%) elapsed=', output.getvalue())
            self.assertIn('batch=4/4 (100.0%) elapsed=', output.getvalue())
            self.assertIn('remaining=00:00:00 finish_local=', output.getvalue())
            self.assertEqual(resumed_ids, full_ids[2:])
            full = torch.load(config.run_dir/'latest.pt', weights_only=True)
            resumed = torch.load(broken.run_dir/'latest.pt', weights_only=True)
            for key in ('model','optimizer','scheduler','scaler','position','rng'):
                self.assert_nested_equal(full[key], resumed[key])
            self.assertEqual(resumed['position'], {'step':8,'epoch':2,'next_batch':0})
            def metrics(path):
                result = []
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    self.assertGreater(event.pop('seconds'), 0)
                    self.assertGreater(event.pop('samples_per_second'), 0)
                    result.append(event)
                return result
            full_metrics = metrics(config.run_dir/'metrics.jsonl')
            self.assertEqual(full_metrics, metrics(broken.run_dir/'metrics.jsonl'))
            self.assertEqual(len(full_metrics), 10)  # Eight batches and two completed epochs.
            self.assertEqual([e['samples'] for e in full_metrics if e['kind']=='epoch'], [8,8])
            with self.assertRaises(FileExistsError):train(broken)
            with self.assertRaises(IdentityMismatch):train(replace(broken,resume=True,batch_size=1))
            (root/'text.txt').write_text('日本語\n')
            with self.assertRaises(IdentityMismatch):train(replace(broken,resume=True))


if __name__ == '__main__':
    unittest.main()
