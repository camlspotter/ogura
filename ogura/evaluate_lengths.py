"""Evaluate saved best.pt on fixed baseline and augmented validation images."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from ogura.text_common import ROOT
from ogura.training.evaluate import evaluate
from ogura.training.metrics import weighted_cer_label
from ogura.training.model import make_model
from ogura.training.render import Vocabulary


def validation_sets(config, vocabulary, paths, identity, all_fonts=False):
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
        if Vocabulary.read(path.parent/'targets.jsonl').characters != vocabulary.source_characters:
            raise ValueError(f'Validation vocabulary differs: {path}')
        # Length groups describe source text, before font-dependent normalization.
        lengths = [len(line) for line in path.read_text(encoding='utf-8').splitlines()]
        conditions = [('baseline', None, None)]
        conditions += ([('augmented', font, western) for font in (config.font, *config.extra_fonts)
                        for western in (config.western_fonts or (None,))]
                       if all_fonts else [('augmented', None, None)])
        for mode, forced_font, forced_western in conditions:
            cfg = replace(config, text=path, limit=None, clean_probability=0)
            if forced_font is not None:
                cfg = replace(cfg, font=forced_font, extra_fonts=(),
                              western_fonts=(forced_western,) if forced_western else ())
            if mode == 'baseline':
                cfg = replace(cfg, extra_fonts=(), western_fonts=(), font_size_min=config.validation_font_size,
                              font_size_max=config.validation_font_size, padding_min=4,
                              padding_max=4, vertical_jitter=0, vertical_full_range=False)
            records, report = prepare_data(cfg, vocabulary)
            if len(records) != manifest['samples'] or not all(manifest['min_length'] <= n <= manifest['max_length'] for n in lengths):
                raise ValueError(f'Validation length/count mismatch: {path}')
            result.append((dict(**(dict(font=forced_font.name + (' + ' + forced_western.name if forced_western else ''),
                                           font_sha256=sha256(forced_font) + (':' + sha256(forced_western) if forced_western else '')) if forced_font else {}),
                                dataset=path.parent.name, mode=mode,
                                text_sha256=expected['validation_text_sha256'],
                                min_length=min(lengths), max_length=max(lengths)),
                           EpochDataset(records, list(range(len(records))), cfg, epoch=0)))
    return result


def evaluate_sets(model, datasets, vocabulary, batch_size, device, evaluation_aliases=None):
    rows = []
    for info, dataset in datasets:
        row = dict(**info, **evaluate(model, dataset, vocabulary, batch_size, device, evaluation_aliases))
        print(f"validation_length dataset={row['dataset']} mode={row['mode']} "
              f"{('font=' + row['font'] + ' ') if 'font' in row else ''}accuracy={row['exact_accuracy']:.2%} CER={row['cer']:.2%}{weighted_cer_label(row)} seconds={row['seconds']:.3f}", flush=True)
        rows.append(row)
    return rows


def validation_font_keys(identity):
    japanese = [identity['font_sha256'], *identity.get('extra_font_sha256', [])]
    western = identity.get('western_font_sha256', [])
    return [j + ':' + w for j in japanese for w in western] if western else japanese


def font_grid_summary(rows, font_hashes):
    """Equal-weight mean of every length/font condition; reject incomplete grids."""
    augmented = [r for r in rows if r.get('mode') == 'augmented' and 'font_sha256' in r]
    def group(row):
        lo, hi = row['min_length'], row['max_length']
        if lo == hi == 5: return 'short5'
        if 20 <= lo <= hi <= 25: return 'normal'
        if lo == hi == 80: return 'long80'
        raise ValueError('Font grid requires lengths 5, 20–25, and 80')
    expected = {(length, font) for length in ('short5', 'normal', 'long80') for font in font_hashes}
    actual = [(group(r), r['font_sha256']) for r in augmented]
    if len(set(font_hashes)) != len(font_hashes) or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate length/font validation grid')
    def average(subset):
        return dict(**({'weighted_cer': math.fsum(r['weighted_cer'] for r in subset)/len(subset)}
                       if all('weighted_cer' in r for r in subset) else {}),
                    cer=math.fsum(r['cer'] for r in subset)/len(subset),
                    exact_accuracy=math.fsum(r['exact_accuracy'] for r in subset)/len(subset),
                    seconds=sum(r['seconds'] for r in subset), conditions=len(subset))
    summaries = []
    for length in ('short5', 'normal', 'long80'):
        summaries.append(dict(kind='validation_length_mean', length=length,
                              **average([r for r in augmented if group(r) == length])))
    for font in font_hashes:
        subset = [r for r in augmented if r['font_sha256'] == font]
        summaries.append(dict(kind='validation_font_mean', font=subset[0]['font'],
                              font_sha256=font, **average(subset)))
    return average(augmented), summaries


def print_grid_summary(overall, summaries):
    for row in summaries:
        label = f"font={row['font']}" if 'font' in row else f"length={row['length']}"
        print(f"{row['kind']} {label} accuracy={row['exact_accuracy']:.2%} CER={row['cer']:.4%}{weighted_cer_label(row)}", flush=True)
    print(f"validation_grid conditions={overall['conditions']} CER={overall['cer']:.4%}{weighted_cer_label(overall)} "
          f"augmented_seconds={overall['seconds']:.3f}", flush=True)


def load_validation_context(checkpoint, run_config=None, font_dir=None):
    """Restore rendering and label configuration, verifying font file identities."""
    from ogura.training.train import TrainConfig, sha256
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    identity = state['identity']
    saved_args = json.loads((run_config or checkpoint.parent/'run_config.json').read_text())['arguments']
    source = state.get('source_characters', state.get('characters'))
    if source is None:
        vocabulary_path = Path(saved_args['vocabulary'])
        if sha256(vocabulary_path) != identity['vocabulary_sha256']:
            raise ValueError('Vocabulary file differs from checkpoint')
        source = Vocabulary.read(vocabulary_path).source_characters
    vocabulary = Vocabulary(source, identity.get('character_aliases'))
    # latest.pt stores architecture and progress in resumable state, unlike best.pt.
    if 'position' in state:
        state = dict(state, channels=identity['settings']['channels'],
                     model_type=identity['settings'].get('model_type', 'small'),
                     epoch=state['position']['epoch'], step=state['position']['step'])
    def font_path(value):
        return font_dir/Path(value).name if font_dir else Path(value)
    config = TrainConfig(**identity['settings'], font=font_path(saved_args['font']),
                         extra_fonts=tuple(font_path(p) for p in saved_args.get('extra_fonts', [])),
                         western_fonts=tuple(font_path(p) for p in saved_args.get('western_fonts', [])))
    if sha256(config.font) != identity['font_sha256'] or [sha256(p) for p in config.extra_fonts] != identity.get('extra_font_sha256', []):
        raise ValueError('Font files differ from checkpoint')
    if [sha256(p) for p in config.western_fonts] != identity.get('western_font_sha256', []):
        raise ValueError('Western font files differ from checkpoint')
    return state, vocabulary, config


def main():
    from ogura.training.train import TrainConfig, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-fonts', action='store_true', help='Evaluate every text in every configured font')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--evaluation-aliases', type=Path, help='Override scoring-only aliases; model labels remain unchanged')
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
    if device.type=='cuda':
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    state, vocabulary, config = load_validation_context(args.checkpoint, args.run_config, args.font_dir)
    identity = state['identity']
    from ogura.training.aliases import CharacterAliases
    evaluation_aliases = (CharacterAliases.read(args.evaluation_aliases) if args.evaluation_aliases
                          else CharacterAliases(identity.get('evaluation_aliases')))
    paths = args.validation_text or [ROOT/'datasets'/name/'validation.txt'
                                    for name in ('validation_short5','validation','validation_long80')]
    all_fonts = args.all_fonts or config.selection_metric == 'mean-font-cer'
    datasets = validation_sets(config, vocabulary, paths, identity, all_fonts=all_fonts)
    model = make_model(len(vocabulary), state['channels'], state.get('model_type', 'small')).to(device)
    model.load_state_dict(state['model'])
    rows = evaluate_sets(model, datasets, vocabulary, args.batch_size, device, evaluation_aliases)
    grid = None
    if all_fonts:
        overall, summaries = font_grid_summary(rows, validation_font_keys(identity))
        print_grid_summary(overall, summaries)
        grid = dict(overall=overall, summaries=summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(dict(checkpoint_sha256=sha256(args.checkpoint), epoch=state['epoch'],
                       step=state['step'], rendering_settings=identity['settings'],
                       character_aliases=vocabulary.aliases.config, evaluation_aliases=evaluation_aliases.config,
                       font_sha256=identity['font_sha256'], extra_font_sha256=identity.get('extra_font_sha256', []),
                       western_font_sha256=identity.get('western_font_sha256', []), results=rows, grid=grid), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f'Report: {args.output}')


if __name__ == '__main__':main()
