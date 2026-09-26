"""Document-level evaluation scope; preserve original datasets and results."""
import json
from pathlib import Path
import re
import hashlib
import shutil


def load_exclusions(path):
    policy = json.loads(Path(path).read_text())
    names = policy['documents']
    if policy['schema_version'] != 1 or not names or len(set(names)) != len(names):
        raise ValueError('Invalid evaluation exclusion policy')
    if any(Path(n).name != n or '\\' in n or not n.endswith('.pdf') for n in names):
        raise ValueError('Exclusions must be PDF basenames')
    return policy


def excluded_image(name, policy):
    if Path(name).name != name or '\\' in name:
        raise ValueError('Expected an image basename')
    match = re.fullmatch(r'(.+)_page-(\d+)_page\.png', name)
    if not match:
        raise ValueError(f'Cannot resolve source document for {name}')
    return match[1] + '.pdf' in policy['documents']


def prepare(source, output, policy_path):
    """Create independent val/test subsets for external training/evaluation code."""
    policy = load_exclusions(policy_path)
    labels = {split: json.loads((source/split/'labels.json').read_text()) for split in ('val', 'test')}
    hashes = {split: hashlib.sha256((source/split/'labels.json').read_bytes()).hexdigest() for split in labels}
    selected = {split: {name: row for name, row in rows.items() if not excluded_image(name, policy)}
                for split, rows in labels.items()}
    if any(not rows for rows in selected.values()):
        raise ValueError('No pages remain in an evaluation split')
    identity = dict(evaluation_scope=policy, source_labels_sha256=hashes,
                    counts={split: len(rows) for split, rows in selected.items()})
    if output.exists():
        saved = json.loads((output/'manifest.json').read_text())
        if saved != dict(status='complete', **identity):
            raise ValueError('Existing evaluation subset differs or is incomplete')
        for split, rows in selected.items():
            if json.loads((output/split/'labels.json').read_text()) != rows:
                raise ValueError('Existing evaluation labels differ')
            for name, row in rows.items():
                if hashlib.sha256((output/split/'images'/name).read_bytes()).hexdigest() != row['img_hash']:
                    raise ValueError(f'Existing evaluation image differs: {name}')
        return identity
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(dict(status='running', **identity), indent=2))
    for split, rows in selected.items():
        (output/split/'images').mkdir(parents=True)
        for name, row in rows.items():
            target = output/split/'images'/name
            shutil.copy2(source/split/'images'/name, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != row['img_hash']:
                raise ValueError(f'Evaluation source checksum differs: {name}')
        (output/split/'labels.json').write_text(json.dumps(rows, ensure_ascii=False))
    (output/'manifest.json').write_text(json.dumps(dict(status='complete', **identity), indent=2))
    return identity


if __name__ == '__main__':
    import argparse
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'outputs/experiment-v1-regenerated')
    parser.add_argument('--output', type=Path, default=root/'outputs/experiment-standard-v2')
    parser.add_argument('--policy', type=Path, default=root/'evaluation_exclusions.json')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(root):
        parser.error('Output must be under ogura/textdet')
    print(json.dumps(prepare(args.source, args.output, args.policy), ensure_ascii=False, indent=2))
