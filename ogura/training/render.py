"""CPU rendering: a list of (text, render parameters) becomes a CTC batch."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import re
import unicodedata
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
    western_font_path: str | None = None


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
    def __init__(self, characters, aliases=None):
        from .aliases import CharacterAliases
        self.aliases = CharacterAliases(aliases)
        self.source_characters = tuple(characters)
        source_set = set(self.source_characters)
        for group in self.aliases.config['groups']:
            if source_set.intersection(group['members']) and group['representative'] not in source_set:
                raise ValueError('Active alias representative must belong to the source vocabulary')
        self.characters = self.source_characters
        if not self.characters or any(len(c) != 1 for c in self.characters):
            raise ValueError('Vocabulary must contain individual Unicode characters')
        if len(set(self.characters)) != len(self.characters):
            raise ValueError('Duplicate vocabulary entries')
        # Vocabulary entries are independent, never a text to compose across boundaries.
        self.characters = tuple(dict.fromkeys(self.aliases.normalize(c) for c in self.characters))
        for pair, composed in self.aliases.compositions.items():
            if set(pair) <= set(self.characters) and self.aliases.normalize(composed) not in self.characters:
                raise ValueError(f'Composed katakana must belong to the vocabulary: {composed}')
        self.ids = {c: i + 1 for i, c in enumerate(self.characters)}
        self.blank = 0

    @classmethod
    def read(cls, path, aliases=None):
        with Path(path).open(encoding='utf-8') as stream:
            return cls((json.loads(line)['character'] for line in stream), aliases)

    def encode(self, text):
        return [self.ids[c] for c in self.aliases.normalize(text)]

    def __len__(self):
        return len(self.characters) + 1


def parameters_for_sample(font_path, seed, epoch, sample_id, size_min=40, size_max=40,
                          extra_fonts=(), padding_min=4, padding_max=4, vertical_jitter=0,
                          clean_probability=0.0, vertical_full_range=False, western_fonts=()):
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
    western = (random.Random(f'{seed}:{epoch}:{sample_id}:western').choice(tuple(map(str, western_fonts)))
               if western_fonts else None)
    return RenderParams(chosen, size, western_font_path=western, padding=padding, vertical_offset=offset,
                        padding_left=rng.randint(padding_min, padding_max),
                        padding_right=rng.randint(padding_min, padding_max),
                        vertical_position=rng.random() if vertical_full_range else None)


def replace_unsupported(text: str, font_path: str) -> str:
    """Replace missing glyphs, collapse U+0020 runs, and trim edge spaces."""
    supported = font_characters(font_path)
    if any(ord(c) not in supported for c in text):
        if ord(' ') not in supported:
            raise ValueError('Font must support the replacement space U+0020')
        text = ''.join(c if ord(c) in supported else ' ' for c in text)
    return re.sub(' +', ' ', text).strip(' \u3000')


def is_western_character(c):
    cp = ord(c)
    return ('0' <= c <= '9' or
            (unicodedata.category(c).startswith('L') and
             (cp <= 0xff or 0x370 <= cp <= 0x52f)))


def western_runs(text, japanese, western):
    """Neutral ASCII punctuation/spaces join Western spans, fullwidth stays Japanese."""
    if not western:
        return [(text, japanese)]
    def eligible(c):
        return (is_western_character(c) or 0x20 <= ord(c) <= 0x7e or
                c in '‘’“”–—… ')
    chunks = []
    for c in text:
        kind = eligible(c)
        if chunks and chunks[-1][0] == kind:
            chunks[-1][1] += c
        else:
            chunks.append([kind, c])
    runs = []
    for eligible_span, span in chunks:
        use_western = eligible_span and any(is_western_character(c) for c in span)
        for c in span:
            path = western if use_western else japanese
            if ord(c) not in font_characters(path) and ord(c) in font_characters(japanese):
                path = japanese
            if runs and runs[-1][1] == path:
                runs[-1] = (runs[-1][0] + c, path)
            else:
                runs.append((c, path))
    return runs


def normalized_text(text, params):
    if not params.western_font_path:
        return replace_unsupported(text, params.font_path)
    normalized = []
    for run, path in western_runs(text, params.font_path, params.western_font_path):
        supported = font_characters(path)
        if ord(' ') not in supported:
            raise ValueError('Font must support replacement space U+0020')
        normalized.append(''.join(c if ord(c) in supported else ' ' for c in run))
    return re.sub(' +', ' ', ''.join(normalized)).strip(' \u3000')


@lru_cache(maxsize=32)
def western_metrics(path):
    """Read horizontal advances and script-specific OpenType kern lookups."""
    with TTFont(path) as font:
        cmap = font.getBestCmap()
        advances = dict(font['hmtx'].metrics)
        units = font['head'].unitsPerEm
        lookups = {}
        if 'GPOS' in font:
            table = font['GPOS'].table
            for script in table.ScriptList.ScriptRecord:
                lang = script.Script.DefaultLangSys
                if lang is None:
                    continue
                indices = []
                for index in lang.FeatureIndex:
                    feature = table.FeatureList.FeatureRecord[index]
                    if feature.FeatureTag == 'kern':
                        indices.extend(feature.Feature.LookupListIndex)
                lookups[script.ScriptTag] = [table.LookupList.Lookup[i] for i in dict.fromkeys(indices)]
        legacy = {}
        if 'kern' in font:
            for subtable in font['kern'].kernTables:
                if subtable.coverage & 1:
                    legacy.update(subtable.kernTable)
        return cmap, advances, units, lookups, legacy


def pair_adjustment(lookups, first, second):
    total = 0
    for lookup in lookups:
        for subtable in lookup.SubTable:
            kind = lookup.LookupType
            if kind == 9:
                kind, subtable = subtable.ExtensionLookupType, subtable.ExtSubTable
            if kind != 2 or first not in subtable.Coverage.glyphs:
                continue
            record = None
            if subtable.Format == 1:
                index = subtable.Coverage.glyphs.index(first)
                record = next((r for r in subtable.PairSet[index].PairValueRecord
                               if r.SecondGlyph == second), None)
            elif subtable.Format == 2:
                record = subtable.Class1Record[subtable.ClassDef1.classDefs.get(first, 0)].Class2Record[
                    subtable.ClassDef2.classDefs.get(second, 0)]
            if record is not None:
                total += getattr(record.Value1, 'XAdvance', 0) if record.Value1 else 0
                break
    return total


@lru_cache(maxsize=4096)
def western_positions(text, path, size):
    """Proportional advances plus kerning, without ligatures or codepoint rewriting."""
    cmap, advances, units, lookups, legacy = western_metrics(path)
    glyphs = [cmap[ord(c)] for c in text]
    positions, cursor = [], 0.
    for i, (c, glyph) in enumerate(zip(text, glyphs)):
        positions.append(cursor)
        step = advances[glyph][0]
        if i + 1 < len(glyphs):
            script = 'grek' if 0x370 <= ord(c) <= 0x3ff else 'cyrl' if 0x400 <= ord(c) <= 0x52f else 'latn'
            selected = lookups.get(script, lookups.get('DFLT', []))
            step += pair_adjustment(selected, glyph, glyphs[i+1]) if selected else legacy.get((glyph, glyphs[i+1]), 0)
        cursor += step * size / units
    return tuple(positions), cursor


def render_mixed(sample):
    p = sample.render_params
    text = normalized_text(sample.text, p)
    if not text:
        raise ValueError("No renderable text remains after edge-space trimming")
    runs = western_runs(text, p.font_path, p.western_font_path)
    for size in range(p.font_size, 0, -1):
        layout = []
        cursor, left, top, right, bottom = 0., 0., 0., 0., 0.
        for run, path in runs:
            western = path == p.western_font_path
            font = load_font(path, size)
            positions, advance = western_positions(run, path, size) if western else ((0.,), font.getlength(run))
            pieces = zip(positions, run) if western else [(0., run)]
            for offset, piece in pieces:
                box = font.getbbox(piece, anchor='ls')
                left = min(left, cursor + offset + box[0])
                right = max(right, cursor + offset + box[2])
                top, bottom = min(top, box[1]), max(bottom, box[3])
                layout.append((cursor + offset, piece, font, {}))
            cursor += advance
        if bottom - top <= p.height - 2*p.padding:
            break
    else:
        raise ValueError('Text cannot fit the requested height')
    pad_left = p.padding if p.padding_left is None else p.padding_left
    pad_right = p.padding if p.padding_right is None else p.padding_right
    width = math.ceil(max(right, cursor) - left) + pad_left + pad_right
    image = Image.new('L', (max(1, width), p.height), 255)
    ink_top = max(p.padding, min(p.height-p.padding-(bottom-top),
                  (p.height-(bottom-top))//2+p.vertical_offset))
    if p.vertical_position is not None:
        if not 0 <= p.vertical_position <= 1:
            raise ValueError('Vertical position must be in [0, 1]')
        available = p.height-2*p.padding-(bottom-top)
        ink_top = p.padding + min(available, int(p.vertical_position*(available+1)))
    draw = ImageDraw.Draw(image)
    for x, run, font, kwargs in layout:
        draw.text((pad_left-left+x, ink_top-top), run, font=font, fill=0, anchor='ls', **kwargs)
    # Narrow proportional glyphs still need enough time positions for CTC repeats.
    minimum_steps = len(text) + sum(a == b for a,b in zip(text, text[1:]))
    minimum_width = max(1, 8*minimum_steps-7)
    if image.width < minimum_width:
        image = image.resize((minimum_width, image.height), Image.Resampling.LANCZOS)
    return image


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
    if p.western_font_path:
        return render_mixed(sample)
    sample = replace(sample, text=normalized_text(sample.text, p))
    if not sample.text:
        raise ValueError("No renderable text remains after edge-space trimming")
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
        samples = [replace(s, text=normalized_text(s.text, s.render_params)) for s in samples]
        if any(not s.text for s in samples):
            raise ValueError('No renderable text remains after missing-glyph replacement and edge-space trimming')
        encoded = [self.vocabulary.encode(s.text) for s in samples]
        images = [render_sample(s) for s in samples]
        # Merging distinct glyphs can introduce repeated CTC labels.
        for i, (image, row) in enumerate(zip(images, encoded)):
            required = len(row) + sum(a == b for a, b in zip(row, row[1:]))
            if (image.width + 7) // 8 < required:
                images[i] = image.resize((8 * required - 7, image.height), Image.Resampling.LANCZOS)
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
            [self.vocabulary.aliases.normalize(s.text) for s in samples], [s.sample_id for s in samples],
        )
