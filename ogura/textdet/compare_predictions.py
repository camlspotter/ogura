"""Compare pretrained and fine-tuned DBNet on local validation pages."""
import argparse
import csv
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def scores(metric):
    recall, precision, iou = metric.summary()
    f1 = (2 * recall * precision / (recall + precision)
          if recall is not None and precision is not None and recall + precision else 0.0)
    return dict(recall=recall, precision=precision, f1=f1, mean_iou=iou,
                matches=metric.matches, ground_truth=metric.num_gts, predictions=metric.num_preds)


def panel(image, gt, predicted, title):
    """Normalized boxes refer to the actual resized/padded model input."""
    result = Image.new('RGB', (image.width, image.height + 36), 'white')
    result.paste(image, (0, 36))
    draw = ImageDraw.Draw(result)
    draw.text((8, 10), title, fill='black')
    for boxes, color, width in ((gt, '#00a040', 3), (predicted, '#e000c0', 1)):
        for box in boxes:
            x0, y0, x1, y1 = np.clip(box[:4], 0, 1)
            coords = (round(x0 * (image.width-1)), round(y0 * (image.height-1))+36,
                      round(x1 * (image.width-1)), round(y1 * (image.height-1))+36)
            draw.rectangle(coords, outline='black', width=width+2)
            draw.rectangle(coords, outline=color, width=width)
    return result


def compare(args):
    import torch
    import doctr
    from doctr import transforms as T
    from doctr.datasets import DetectionDataset
    from doctr.models import detection
    from doctr.utils.metrics import LocalizationConfusion
    from torchvision.transforms.v2 import Normalize
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for a local smoke test')
    if args.amp and not args.device.startswith('cuda'):
        raise ValueError('--amp requires a CUDA device')
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    dataset = DetectionDataset(str(args.data / 'images'), str(args.data / 'labels.json'),
                               sample_transforms=T.Resize((args.input_size, args.input_size),
                                                          preserve_aspect_ratio=True, symmetric_pad=True))
    if dataset.class_names != ['words']:
        raise ValueError(f'Expected single-class labels: {dataset.class_names}')
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    if not count:
        raise ValueError('Empty validation dataset')
    args.output.mkdir(parents=True)
    report = dict(status='running', data=str(args.data.resolve()), checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  labels_sha256=hashlib.sha256((args.data/'labels.json').read_bytes()).hexdigest(),
                  input_size=args.input_size, amp=args.amp, device=args.device, iou_threshold=0.5,
                  partial_run=count < len(dataset), pages=count, coordinate_space='normalized resized/padded input',
                  versions=dict(torch=torch.__version__, doctr=doctr.__version__), models={})
    def save():
        (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    save()
    normalize = Normalize(mean=(.798, .785, .772), std=(.264, .2749, .287))
    rows, predictions = [], {}
    try:
        for kind in ('pretrained', 'finetuned'):
            model = detection.db_resnet34(pretrained=kind == 'pretrained', pretrained_backbone=False,
                                         assume_straight_pages=True, class_names=dataset.class_names)
            if kind == 'finetuned':
                model.load_state_dict(checkpoint, strict=True)
            model = model.to(args.device).eval()
            total = LocalizationConfusion(iou_thresh=.5)
            predictions[kind] = []
            for index in range(count):
                image, target = dataset[index]
                with torch.inference_mode(), (torch.amp.autocast('cuda') if args.amp else nullcontext()):
                    output = model(normalize(image.unsqueeze(0).to(args.device)), return_preds=True)
                boxes = output['preds'][0]['words']
                gt = target['words']
                total.update(gt, boxes[:, :4])
                metric = LocalizationConfusion(iou_thresh=.5)
                metric.update(gt, boxes[:, :4])
                name = dataset.data[index][0]
                rows.append(dict(image=name, model=kind, **scores(metric)))
                predictions[kind].append(boxes.tolist())
                print(f'{kind}: {index+1}/{count} {name}', flush=True)
            report['models'][kind] = scores(total)
            save()
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
        for index in range(count):
            tensor, target = dataset[index]
            image = Image.fromarray((tensor.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype('uint8'))
            gt = target['words']
            panels = [panel(image, gt, [], 'Ground truth (green)')]
            for kind in ('pretrained', 'finetuned'):
                panels.append(panel(image, gt, predictions[kind][index], f'{kind}: GT green / prediction magenta'))
            combined = Image.new('RGB', (image.width*3, image.height+36), 'white')
            name = Path(dataset.data[index][0]).stem
            for i, view in enumerate(panels):
                combined.paste(view, (i*image.width, 0))
                view.save(args.output/f'{name}_{("gt", "pretrained", "finetuned")[i]}.png')
            combined.save(args.output/f'{name}_comparison.png')
        with (args.output/'pages.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output/'predictions.json').write_text(json.dumps(dict(
            images=[dataset.data[i][0] for i in range(count)], predictions=predictions)))
        report['status'] = 'complete'
        save()
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        save()
        raise
    print(json.dumps(report['models'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('ogura/textdet/outputs/experiment-v1-regenerated/val'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('ogura/textdet/outputs/validation-comparison-v1'))
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.input_size < 32 or args.input_size % 32 or (args.limit is not None and args.limit < 1):
        parser.error('input-size must be a positive multiple of 32; limit must be positive')
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parent):
        parser.error('Output must be under ogura/textdet')
    compare(args)


if __name__ == '__main__':
    main()
