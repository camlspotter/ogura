"""Download pinned Noto CJK Japanese fonts for the first augmentation stage."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from ogura.text_common import ROOT

REVISION = 'f8d157532fbfaeda587e826d4cd5b21a49186f7c'
BASE = f'https://raw.githubusercontent.com/notofonts/noto-cjk/{REVISION}'


def download(output):
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for family in ('Sans', 'Serif'):
        for weight in ('Regular', 'Bold'):
            name = f'Noto{family}CJKjp-{weight}.otf'
            entries.append((name, f'{family}/OTF/Japanese/{name}'))
        entries.append((f'LICENSE-Noto{family}', f'{family}/LICENSE'))
    manifest = {'revision': REVISION, 'files': []}
    for name, source in entries:
        target = output / name
        url = f'{BASE}/{source}'
        # Fetch to a temporary file; never overwrite a different existing font.
        temporary = output / (name + '.download')
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            digest = hashlib.sha256(data).hexdigest()
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'{target} differs from the pinned source; choose a new --output directory')
            else:
                temporary.write_bytes(data)
                temporary.replace(target)
            manifest['files'].append(dict(name=name, url=url, sha256=digest))
            print(target, flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    temporary = output / 'noto-fonts.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(output / 'noto-fonts.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'corpus/fonts')
    download(parser.parse_args().output)


if __name__ == '__main__':
    main()
