"""Save DBNet probability maps and compare postprocessing on selected pages or a full dataset."""
import argparse
import csv
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .compare_predictions import panel, scores

ROOT = Path(__file__).resolve().parent
PAGES = [f'public_document_ministry01376_page-{page:04d}_page.png' for page in (6, 11)]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sweep(probability, gt, image, output, thresholds, ratios, box_threshold):
    import cv2
    from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
    from doctr.utils.metrics import LocalizationConfusion

    rows, predictions = [], {}
    for threshold in thresholds:
        processor = DBPostProcessor(bin_thresh=threshold, box_thresh=box_threshold,
                                    assume_straight_pages=True)
        binary = (probability >= threshold).astype('uint8')
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, processor._opening_kernel)
        Image.fromarray(binary * 255).save(output/f'binary-{threshold:g}.png')
        Image.fromarray(opened * 255).save(output/f'opened-{threshold:g}.png')
        for ratio in ratios:
            processor.unclip_ratio = ratio
            boxes = processor(probability[None, ..., None])[0][0]
            metric = LocalizationConfusion(iou_thresh=.5)
            metric.update(gt, boxes[:, :4])
            tag = f'bin-{threshold:g}_unclip-{ratio:g}'
            predictions[tag] = boxes.tolist()
            panel(image, gt, boxes, tag + ' / GT green / predicted magenta').save(output/f'{tag}.png')
            rows.append(dict(bin_thresh=threshold, unclip_ratio=ratio, box_thresh=box_threshold,
                             **scores(metric)))
    (output/'predictions.json').write_text(json.dumps(predictions))
    return rows


def aggregate(rows):
    """Micro-average counts across pages; never average per-page F1."""
    groups = {}
    for row in rows:
        key = (row['model'], row['bin_thresh'], row['unclip_ratio'], row['box_thresh'])
        result = groups.setdefault(key, dict(zip(('model','bin_thresh','unclip_ratio','box_thresh'), key),
                                            pages=0, matches=0, ground_truth=0, predictions=0))
        result['pages'] += 1
        for name in ('matches', 'ground_truth', 'predictions'):
            result[name] += row[name]
    for result in groups.values():
        m, g, p = (result[k] for k in ('matches', 'ground_truth', 'predictions'))
        result.update(recall=m/g if g else 0., precision=m/p if p else 0.,
                      f1=2*m/(g+p) if g+p else 0.)
    return sorted(groups.values(), key=lambda r: (-r['f1'], r['model'], r['bin_thresh'], r['unclip_ratio']))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    import torch
    import doctr
    from doctr import transforms as T
    from doctr.datasets import DetectionDataset
    from doctr.models import detection
    from torchvision.transforms.v2 import Normalize

    if args.amp and not args.device.startswith('cuda'):
        raise ValueError('--amp requires CUDA')
    dataset = DetectionDataset(str(args.data/'images'), str(args.data/'labels.json'),
                               sample_transforms=T.Resize((args.input_size, args.input_size),
                                                          preserve_aspect_ratio=True, symmetric_pad=True))
    if dataset.class_names != ['words']:
        raise ValueError('Expected words labels')
    inventory = {item[0]: i for i, item in enumerate(dataset.data)}
    images = list(inventory) if args.all_pages else (args.images or PAGES)
    if not images or len(images) != len(set(images)):
        raise ValueError('Expected a nonempty list of unique pages')
    exclusions = None
    excluded = []
    if getattr(args, 'exclude_documents', None):
        from .evaluation_scope import load_exclusions, excluded_image
        exclusions = load_exclusions(args.exclude_documents)
        excluded = [name for name in images if excluded_image(name, exclusions)]
        images = [name for name in images if name not in excluded]
        if not images:
            raise ValueError('No evaluation pages remain after exclusions')
    if args.expected_pages is not None and len(images) != args.expected_pages:
        raise ValueError(f'Expected {args.expected_pages} pages, got {len(images)}')
    if any(name not in inventory for name in images):
        raise ValueError('Requested page missing from dataset')
    checkpoints = {'synth7000': args.previous, 'synth9000': args.current}
    if args.model != 'both':
        checkpoints = {args.model: checkpoints[args.model]}
    names = {'synth7000': getattr(args, 'previous_name', 'synth7000'),
             'synth9000': getattr(args, 'current_name', 'synth9000')}
    if len(set(names.values())) != 2 or any(not name or not name.isascii() or
            not all(c.isalnum() or c in '_-' for c in name) for name in names.values()):
        raise ValueError('Model names must be distinct safe directory names')
    checkpoints = {names[name]: path for name, path in checkpoints.items()}
    hashes = {name: sha(path) for name, path in checkpoints.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', input_size=args.input_size, images=images, pages=len(images),
                  all_pages=args.all_pages, data=str(args.data.resolve()),
                  labels_sha256=sha(args.data/'labels.json'), checkpoints=hashes,
                  bin_thresholds=args.thresholds, unclip_ratios=args.ratios, box_thresh=args.box_threshold,
                  amp=args.amp, device=args.device, versions=dict(torch=torch.__version__, doctr=doctr.__version__),
                  note='unclip=0 means no polygon expansion, not no postprocessing; select settings on validation only')
    rows = []
    if exclusions:
        report['evaluation_scope'] = exclusions
        report['excluded_images'] = excluded
        report['exclusions_sha256'] = sha(args.exclude_documents)
    normalize = Normalize(mean=(.798, .785, .772), std=(.264, .2749, .287))
    try:
        for name, checkpoint in checkpoints.items():
            model = detection.db_resnet34(pretrained=False, pretrained_backbone=False,
                                          assume_straight_pages=True, class_names=['words'])
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            model = model.to(args.device).eval()
            for filename in images:
                tensor, target = dataset[inventory[filename]]
                with torch.inference_mode(), (torch.amp.autocast('cuda') if args.amp else nullcontext()):
                    result = model(normalize(tensor.unsqueeze(0).to(args.device)), return_model_output=True)
                probability = result['out_map'][0, 0].detach().float().cpu().numpy()
                dest = args.output/name/Path(filename).stem
                dest.mkdir(parents=True)
                np.save(dest/'probability.npy', probability)
                Image.fromarray((probability * 255).round().astype('uint8')).save(dest/'probability.png')
                image = Image.fromarray((tensor.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype('uint8'))
                image.save(dest/'input.png')
                gt = target['words']
                (dest/'ground_truth.json').write_text(json.dumps(gt.tolist()))
                for row in sweep(probability, gt, image, dest, args.thresholds, args.ratios, args.box_threshold):
                    rows.append(dict(model=name, image=filename, **row))
                print(f'{name}: {filename} complete', flush=True)
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
        write_csv(args.output/'metrics.csv', rows)
        report['aggregate'] = aggregate(rows)
        write_csv(args.output/'ranking.csv', report['aggregate'])
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'outputs/experiment-v1-regenerated/val')
    parser.add_argument('--previous', type=Path, default=ROOT/'outputs/db-resnet34-synth7000-v1/synth7000-db-resnet34-v1.pt')
    parser.add_argument('--current', type=Path, default=ROOT/'outputs/db-resnet34-synth9000-v1/synth9000-db-resnet34-v1.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/postprocess-diagnosis-v1')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--images', nargs='+')
    selection.add_argument('--all-pages', action='store_true')
    parser.add_argument('--expected-pages', type=int)
    parser.add_argument('--exclude-documents', type=Path,
                        help='Explicit document exclusion policy; original dataset is retained')
    parser.add_argument('--model', choices=['both','synth7000','synth9000'], default='both')
    parser.add_argument('--previous-name', default='synth7000')
    parser.add_argument('--current-name', default='synth9000')
    parser.add_argument('--input-size', type=int, default=1536)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.2, .3, .4, .5])
    parser.add_argument('--ratios', type=float, nargs='+', default=[0, 1, 1.5])
    parser.add_argument('--box-threshold', type=float, default=.1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    if args.input_size < 32 or args.input_size % 32:
        parser.error('input-size must be a positive multiple of 32')
    if not all(0 < t < 1 for t in args.thresholds) or not all(np.isfinite(r) and r >= 0 for r in args.ratios):
        parser.error('Invalid thresholds or unclip ratios')
    if not 0 <= args.box_threshold <= 1:
        parser.error('Invalid box threshold')
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under ogura/textdet')
    run(args)


if __name__ == '__main__':
    main()
