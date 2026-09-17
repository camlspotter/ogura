"""CPU rendering: a list of (text, render parameters) becomes a CTC batch."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import re
from pathlib import Path
import random

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import torch


@lru_cache(maxsize=32)
def font_characters(path: str) -> frozenset[int]:
    with TTFont(path) as font:
        return frozenset(cp for cp, glyph in font.getBestCmap().items() if glyph != '.notdef')


@lru_cache(maxsize=128)
def load_font(path: str, size: int):
    # BASIC avoids depending on the optional RAQM shaping library.
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)


@dataclass(frozen=True)
class RenderParams:
    font_path: str
    font_size: int = 40
    height: int = 48
    padding: int = 4
    vertical_offset: int = 0
    padding_left: int | None = None
    padding_right: int | None = None
    vertical_position: float | None = None


@dataclass(frozen=True)
class Sample:
    text: str
    render_params: RenderParams
    sample_id: str = ''


@dataclass
class Batch:
    images: torch.Tensor
    image_widths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    texts: list[str]
    sample_ids: list[str]


class Vocabulary:
    def __init__(self, characters):
        self.characters = tuple(characters)
        if not self.characters or any(len(c) != 1 for c in self.characters):
            raise ValueError('Vocabulary must contain individual Unicode characters')
        if len(set(self.characters)) != len(self.characters):
            raise ValueError('Duplicate vocabulary entries')
        self.ids = {c: i + 1 for i, c in enumerate(self.characters)}
        self.blank = 0

    @classmethod
    def read(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            return cls(json.loads(line)['character'] for line in stream)

    def encode(self, text):
        return [self.ids[c] for c in text]

    def __len__(self):
        return len(self.characters) + 1


def parameters_for_sample(font_path, seed, epoch, sample_id, size_min=40, size_max=40,
                          extra_fonts=(), padding_min=4, padding_max=4, vertical_jitter=0,
                          clean_probability=0.0, vertical_full_range=False):
    """No worker/global RNG dependency; replay gives the same render parameters."""
    if not 1 <= size_min <= size_max:
        raise ValueError('Invalid font size range')
    digest = hashlib.sha256(f'{seed}:{epoch}:{sample_id}'.encode()).digest()
    rng = random.Random(int.from_bytes(digest, 'big'))
    if not 0 <= padding_min <= padding_max < 24 or vertical_jitter < 0:
        raise ValueError('Invalid padding or vertical jitter')
    if not 0 <= clean_probability <= 1:
        raise ValueError('Invalid clean probability')
    if clean_probability and rng.random() < clean_probability:
        return RenderParams(str(font_path))
    size = rng.randint(size_min, size_max)
    chosen = rng.choice((str(font_path), *(str(p) for p in extra_fonts))) if extra_fonts else str(font_path)
    padding = rng.randint(padding_min, padding_max)
    offset = rng.randint(-vertical_jitter, vertical_jitter)
    return RenderParams(chosen, size, padding=padding, vertical_offset=offset,
                        padding_left=rng.randint(padding_min, padding_max),
                        padding_right=rng.randint(padding_min, padding_max),
                        vertical_position=rng.random() if vertical_full_range else None)


def replace_unsupported(text: str, font_path: str) -> str:
    """Replace missing glyphs, then collapse U+0020 runs without stripping edges."""
    supported = font_characters(font_path)
    if any(ord(c) not in supported for c in text):
        if ord(' ') not in supported:
            raise ValueError('Font must support the replacement space U+0020')
        text = ''.join(c if ord(c) in supported else ' ' for c in text)
    return re.sub(' +', ' ', text)


def render_sample(sample: Sample) -> Image.Image:
    p = sample.render_params
    if not sample.text or any(c in sample.text for c in '\r\n\x85\u2028\u2029'):
        raise ValueError('Expected a nonempty single-line text')
    if p.height < 1 or p.padding < 0 or 2 * p.padding >= p.height or p.font_size < 1:
        raise ValueError('Invalid rendering dimensions')
    pad_left = p.padding if p.padding_left is None else p.padding_left
    pad_right = p.padding if p.padding_right is None else p.padding_right
    if pad_left < 0 or pad_right < 0:
        raise ValueError('Horizontal padding must be nonnegative')
    sample = replace(sample, text=replace_unsupported(sample.text, p.font_path))
    for size in range(p.font_size, 0, -1):
        font = load_font(p.font_path, size)
        left, top, right, bottom = font.getbbox(sample.text)
        if bottom - top <= p.height - 2 * p.padding:
            break
    else:
        raise ValueError('Text cannot fit the requested height')
    width = math.ceil(max(right, font.getlength(sample.text)) - min(0, left)) + pad_left + pad_right
    image = Image.new('L', (max(1, width), p.height), 255)
    ink_top = (p.height - (bottom - top)) // 2
    ink_top = max(p.padding, min(p.height - p.padding - (bottom - top), ink_top + p.vertical_offset))
    if p.vertical_position is not None:
        if not 0 <= p.vertical_position <= 1:
            raise ValueError('Vertical position must be in [0, 1]')
        available = p.height - 2 * p.padding - (bottom - top)
        ink_top = p.padding + min(available, int(p.vertical_position * (available + 1)))
    ImageDraw.Draw(image).text(
        (pad_left - min(0, left), ink_top - top),
        sample.text, font=font, fill=0,
    )
    return image


class BatchRenderer:
    """Picklable collator; samples carry all resolved rendering parameters."""
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, samples: list[Sample]) -> Batch:
        if not samples or any(s.render_params.height != 48 for s in samples):
            raise ValueError('Expected a nonempty batch of height-48 images')
        if any(not s.text or any(c in s.text for c in '\r\n\x85\u2028\u2029') for s in samples):
            raise ValueError('Expected nonempty single-line text')
        samples = [replace(s, text=replace_unsupported(s.text, s.render_params.font_path)) for s in samples]
        encoded = [self.vocabulary.encode(s.text) for s in samples]
        images = [render_sample(s) for s in samples]
        widths = [im.width for im in images]
        max_width = math.ceil(max(widths) / 8) * 8
        tensor = torch.ones(len(samples), 1, 48, max_width, dtype=torch.float32)
        for index, image in enumerate(images):
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            tensor[index, 0, :, :image.width] = pixels.reshape(48, image.width).float() / 255
        return Batch(
            tensor, torch.tensor(widths, dtype=torch.int64),
            torch.tensor([token for row in encoded for token in row], dtype=torch.int64),
            torch.tensor([len(row) for row in encoded], dtype=torch.int64),
            [s.text for s in samples], [s.sample_id for s in samples],
        )
