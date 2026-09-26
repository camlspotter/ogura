"""Move recognition assets into textrec; keep shared corpus at repo/corpus.

No symlinks are created. Existing compatibility links are removed only after
checking their targets. Manifests/checkpoints are not rewritten.
"""
from pathlib import Path
import argparse

ASSETS = ('datasets', 'runs', 'cache', 'charset', 'previews')


def migrate(repo):
    project = repo / 'textrec'
    actions = []
    pairs = [(repo/name, project/name) for name in ASSETS]
    pairs.append((project/'corpus', repo/'corpus'))
    for source, target in pairs:
        if source.is_symlink():
            if source.resolve() != target.resolve():
                raise ValueError(f'Unexpected link: {source}')
            actions.append(('unlink', source, target)); continue
        if target.is_symlink():
            if target.resolve() != source.resolve() or not source.is_dir():
                raise ValueError(f'Unexpected link: {target}')
            actions.append(('unlink', target, source))
        elif source.exists() and target.exists():
            raise FileExistsError(f'Refusing to merge: {source}, {target}')
        if source.exists():
            if not source.is_dir(): raise ValueError(f'Expected directory: {source}')
            actions.append(('move', source, target))
    for action, source, target in actions:
        if action == 'unlink': source.unlink()
        else: source.rename(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    migrate(parser.parse_args().repo.resolve())
