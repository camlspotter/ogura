"""Download pinned training fonts, optionally including light and rounded faces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from ogura.text_common import ROOT

CATALOG = Path(__file__).with_name('training_fonts.json')


def entries(include_rounded=False):
    catalog = json.loads(CATALOG.read_text())
    return catalog['noto'] + (catalog['rounded'] if include_rounded else [])


def font_paths(output, include_rounded=False):
    return [Path(output)/e['name'] for e in entries(include_rounded)
            if e['name'].endswith(('.otf', '.ttf'))]


def fetch(entry, output):
    target = output/entry['name']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'{target} differs from the pinned source; choose a new --output directory')
        return target
    with urllib.request.urlopen(entry['url'], timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError(f"Downloaded checksum mismatch: {entry['name']}")
    with tempfile.NamedTemporaryFile(dir=output, prefix='.font-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        # Exclusive creation also prevents a concurrent download from overwriting a file.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def download(output, include_rounded=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = entries(include_rounded)
    for entry in selected:
        print(fetch(entry, output), flush=True)
    manifest = output/('training-fonts.json' if include_rounded else 'noto-fonts.json')
    with tempfile.NamedTemporaryFile(mode='w', dir=output, prefix='.manifest-', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(dict(files=selected), stream, indent=2)
        stream.write('\n')
    try:
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'corpus/fonts')
    parser.add_argument('--include-rounded', action='store_true',
                        help='Also fetch Noto Sans Light and Zen Maru Gothic Light/Regular')
    args = parser.parse_args()
    download(args.output, args.include_rounded)


if __name__ == '__main__':
    main()
