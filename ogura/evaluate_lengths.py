"""Evaluate saved best.pt on fixed baseline and augmented validation images."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from ogura.text_common import ROOT
from ogura.training.evaluate import evaluate
from ogura.training.model import LineCNN
from ogura.training.render import Vocabulary


def validation_sets(config, vocabulary, paths, identity):
    from ogura.training.train import EpochDataset, prepare_data, sha256
    result = []
    for path in paths:
        path = Path(path)
        manifest = json.loads((path.parent/'manifest.json').read_text())
        expected = dict(training_text_sha256=identity['text_sha256'],
                        targets_sha256=identity['vocabulary_sha256'],
                        validation_text_sha256=sha256(path), split='validation')
        if any(manifest.get(k) != value for k,value in expected.items()):
            raise ValueError(f'Validation manifest does not match checkpoint: {path}')
        if Vocabulary.read(path.parent/'targets.jsonl').characters != vocabulary.characters:
            raise ValueError(f'Validation vocabulary differs: {path}')
        for mode in ('baseline', 'augmented'):
            cfg = replace(config, text=path, limit=None, clean_probability=0)
            if mode == 'baseline':
                cfg = replace(cfg, extra_fonts=(), font_size_min=config.validation_font_size,
                              font_size_max=config.validation_font_size, padding_min=4,
                              padding_max=4, vertical_jitter=0, vertical_full_range=False)
            records, report = prepare_data(cfg, vocabulary)
            lengths = [len(text) for text,_ in records]
            if len(records) != manifest['samples'] or not all(manifest['min_length'] <= n <= manifest['max_length'] for n in lengths):
                raise ValueError(f'Validation length/count mismatch: {path}')
            result.append((dict(dataset=path.parent.name, mode=mode,
                                text_sha256=expected['validation_text_sha256'],
                                min_length=min(lengths), max_length=max(lengths)),
                           EpochDataset(records, list(range(len(records))), cfg, epoch=0)))
    return result


def evaluate_sets(model, datasets, vocabulary, batch_size, device):
    rows = []
    for info, dataset in datasets:
        row = dict(**info, **evaluate(model, dataset, vocabulary, batch_size, device))
        print(f"validation_length dataset={row['dataset']} mode={row['mode']} "
              f"accuracy={row['exact_accuracy']:.2%} CER={row['cer']:.2%} seconds={row['seconds']:.3f}", flush=True)
        rows.append(row)
    return rows


def main():
    from ogura.training.train import TrainConfig, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--font-dir', type=Path, help='Override font directory after moving a run')
    parser.add_argument('--validation-text', type=Path, action='append')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; never overwrite')
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.batch_size < 1 or args.threads < 1:raise ValueError('Batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type not in ('cpu','cuda'):raise ValueError('Use CPU or CUDA')
    if device.type=='cuda':torch.cuda.set_device(device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    identity = state['identity']
    vocabulary = Vocabulary(state['characters'])
    saved_args = json.loads((args.run_config or args.checkpoint.parent/'run_config.json').read_text())['arguments']
    def font_path(value):
        return args.font_dir/Path(value).name if args.font_dir else Path(value)
    config = TrainConfig(**identity['settings'], font=font_path(saved_args['font']),
                         extra_fonts=tuple(font_path(p) for p in saved_args.get('extra_fonts', [])))
    if sha256(config.font) != identity['font_sha256'] or [sha256(p) for p in config.extra_fonts] != identity.get('extra_font_sha256', []):
        raise ValueError('Font files differ from checkpoint')
    paths = args.validation_text or [ROOT/'datasets'/name/'validation.txt'
                                    for name in ('validation_short5','validation','validation_long80')]
    datasets = validation_sets(config, vocabulary, paths, identity)
    model = LineCNN(len(vocabulary), state['channels']).to(device)
    model.load_state_dict(state['model'])
    rows = evaluate_sets(model, datasets, vocabulary, args.batch_size, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(dict(checkpoint_sha256=sha256(args.checkpoint), epoch=state['epoch'],
                       step=state['step'], rendering_settings=identity['settings'],
                       font_sha256=identity['font_sha256'], extra_font_sha256=identity.get('extra_font_sha256', []),
                       results=rows), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'Report: {args.output}')


if __name__ == '__main__':main()
