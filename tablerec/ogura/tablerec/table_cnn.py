"""Two-channel U-Net and in-memory table boundary training data."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from .synth_table_cells import TableCellDataset

CHANNELS = ('horizontal', 'vertical')


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        layers = []
        for source in (incoming, outgoing):
            layers.extend([nn.Conv2d(source, outgoing, 3, padding=1, bias=False),
                           nn.GroupNorm(math.gcd(8, outgoing), outgoing), nn.ReLU(inplace=True)])
        super().__init__(*layers)


class TableUNet(nn.Module):
    """Return logits [N, 2, H, W]; horizontal then vertical, independent sigmoid."""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        if base_channels < 4:
            raise ValueError('base_channels must be at least 4')
        self.base_channels = base_channels
        widths = [base_channels*x for x in (1, 2, 4, 8, 8)]
        self.encoder = nn.ModuleList([ConvBlock(a, b) for a, b in zip([3]+widths[:-1], widths)])
        self.decoder = nn.ModuleList([ConvBlock(a+b, b) for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.output = nn.Conv2d(widths[0], 2, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError('Expected nonempty N x 3 x H x W input')
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 16, 0, (-height) % 16), value=1)
        skips = []
        for index, block in enumerate(self.encoder):
            if index:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.output(x)[..., :height, :width]


def image_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)).div_(255)


class TableTrainingDataset(Dataset):
    def __init__(self, root: Path, *, font_dir: Path | None = None, limit: int | None = None):
        self.source = TableCellDataset(root, font_dir=font_dir)
        if limit is not None and limit <= 0:
            raise ValueError('Dataset limit must be positive')
        self.count = min(len(self.source), limit) if limit is not None else len(self.source)
        if self.count == 0:
            raise ValueError('Dataset is empty')
        self.sizes, self.seeds = [], set()
        for record in self.source.records[:self.count]:
            recipe = json.loads((self.source.root/record['recipe']).read_text())
            self.sizes.append((recipe['height'], recipe['width']))
            self.seeds.add(recipe['seed'])

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        sample = self.source[index]
        target = torch.from_numpy(np.stack([np.array(sample[c], dtype=np.float32) for c in CHANNELS])).div_(255)
        return dict(image=image_tensor(sample['image']), target=target, id=sample['id'])


class SizeBatchSampler(Sampler):
    """Shuffle, locally sort by shape, then shuffle batches to reduce padding."""
    def __init__(self, sizes, batch_size: int, *, seed: int = 0, shuffle: bool = True):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.sizes, self.batch_size = sizes, batch_size
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def __len__(self):
        return math.ceil(len(self.sizes)/self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        indices = torch.randperm(len(self.sizes), generator=generator).tolist() if self.shuffle else list(range(len(self.sizes)))
        batches = []
        window = self.batch_size*32
        for start in range(0, len(indices), window):
            pool = sorted(indices[start:start+window], key=lambda i: (round(math.log2(self.sizes[i][1]/self.sizes[i][0])*2), self.sizes[i][0]*self.sizes[i][1]))
            batches.extend(pool[i:i+self.batch_size] for i in range(0, len(pool), self.batch_size))
        order = torch.randperm(len(batches), generator=generator).tolist() if self.shuffle else range(len(batches))
        yield from (batches[i] for i in order)


def collate_tables(samples):
    height = math.ceil(max(s['image'].shape[1] for s in samples)/16)*16
    width = math.ceil(max(s['image'].shape[2] for s in samples)/16)*16
    images = torch.ones(len(samples), 3, height, width)
    targets = torch.zeros(len(samples), 2, height, width)
    valid = torch.zeros(len(samples), 1, height, width)
    sizes = []
    for i, sample in enumerate(samples):
        h, w = sample['image'].shape[-2:]
        images[i, :, :h, :w] = sample['image']
        targets[i, :, :h, :w] = sample['target']
        valid[i, :, :h, :w] = 1
        sizes.append((h, w))
    return dict(image=images, target=targets, valid=valid, sizes=sizes, ids=[s['id'] for s in samples])


def boundary_loss(logits, target, valid):
    """Per-table/channel BCE + soft Dice; padding excluded from both terms."""
    axes = (-2, -1)
    area = valid.sum(axes).clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*valid).sum(axes)/area
    probability = logits.sigmoid()
    intersection = (probability*target*valid).sum(axes)
    denominator = ((probability+target)*valid).sum(axes)
    dice_loss = 1-(2*intersection+1e-6)/(denominator+1e-6)
    return (bce+dice_loss).mean()


def boundary_metrics(logits, target, valid):
    """Per-channel means over tables. Threshold F1 is secondary to soft metrics."""
    p = logits.sigmoid()
    axes = (-2, -1)
    area = valid.sum(axes).clamp_min(1)
    mae = ((p-target).abs()*valid).sum(axes)/area
    dice = (2*(p*target*valid).sum(axes)+1e-6)/(((p+target)*valid).sum(axes)+1e-6)
    pred, truth = (p >= 0.5).float(), (target >= 0.5).float()
    tp = (pred*truth*valid).sum(axes)
    precision = (tp+1e-6)/((pred*valid).sum(axes)+1e-6)
    recall = (tp+1e-6)/((truth*valid).sum(axes)+1e-6)
    f1 = (2*tp+1e-6)/(((pred+truth)*valid).sum(axes)+1e-6)
    values = dict(mae=mae, soft_dice=dice, precision_at_05=precision, recall_at_05=recall, f1_at_05=f1)
    return {f'{channel}/{name}': value[:, i].mean().item() for name, value in values.items() for i, channel in enumerate(CHANNELS)}


@torch.inference_mode()
def predict_image(model: TableUNet, image: Image.Image, device='cpu') -> torch.Tensor:
    """Return CPU probabilities [2,H,W], without saving images or changing model mode."""
    training = model.training
    model.eval()
    try:
        return model(image_tensor(image).unsqueeze(0).to(device))[0].sigmoid().cpu()
    finally:
        model.train(training)


def load_model(path: Path, device='cpu') -> TableUNet:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('channels') != list(CHANNELS):
        raise ValueError('Unsupported table CNN checkpoint')
    model = TableUNet(checkpoint['base_channels'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()
