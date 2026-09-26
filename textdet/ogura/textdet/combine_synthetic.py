"""Copy text-only and table-page images into an independent docTR training set."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def combine(text, tables, output, text_count=5000, table_count=2000, headings=None, heading_tables=None, heading_count=1000,
            image_text=None, image_tables=None, image_text_count=1334, image_table_count=666):
    if output.exists():
        raise FileExistsError(output)
    sources = []
    specs = [('text',text,text_count,False), ('table',tables,table_count,True)]
    if (headings is None) != (heading_tables is None):
        raise ValueError('Both heading datasets are required')
    if headings is not None:
        specs += [('heading',headings,heading_count,False), ('heading-table',heading_tables,heading_count,True)]
    if (image_text is None) != (image_tables is None):
        raise ValueError('Both image datasets are required')
    if image_text is not None:
        specs += [('image',image_text,image_text_count,False), ('image-table',image_tables,image_table_count,True)]
    total = sum(item[2] for item in specs)
    for prefix, source, expected, has_tables in specs:
        manifest = json.loads((source/'manifest.json').read_text())
        labels = json.loads((source/'labels.json').read_text())
        if prefix.startswith('heading') and not all(manifest.get('settings', {}).get(k) == v
                for k,v in dict(section_headings=True, colored_text=True, title_style='mixed').items()):
            raise ValueError(f'{source}: heading decoration settings missing')
        pages = manifest.get('pages', [])
        if manifest.get('status') != 'complete' or len(pages) != expected or len(labels) != expected:
            raise ValueError(f'{source}: expected {expected} complete pages')
        names = [p['image'] for p in pages]
        if len(set(names)) != expected or set(names) != set(labels):
            raise ValueError(f'{source}: manifest and label inventory differ')
        if any(bool(p.get('table')) != has_tables for p in pages):
            raise ValueError(f'{source}: wrong page type for {prefix} set')
        if prefix.startswith('image'):
            settings = manifest.get('settings', {})
            if not settings.get('image_assets') or settings.get('image_float_fraction') != .5:
                raise ValueError(f'{source}: expected reviewed image assets and half float layouts')
            if any(p.get('image_asset', {}).get('layout') not in ('float', 'block') for p in pages):
                raise ValueError(f'{source}: missing image layout metadata')
            if sum(p['image_asset']['layout'] == 'float' for p in pages) != round(expected * .5):
                raise ValueError(f'{source}: wrong float layout count')
        for name in names:
            if Path(name).name != name or not (source/'images'/name).is_file():
                raise ValueError(f'{source}: invalid or missing image {name}')
        sources.append((prefix, source, labels, sha(source/'manifest.json'), sha(source/'labels.json')))
    output.mkdir(parents=True)
    (output/'images').mkdir()
    report = dict(status='running', role='synthetic_training', pages=[], sources=[])
    combined = {}
    try:
        for prefix, source, labels, manifest_hash, labels_hash in sources:
            report['sources'].append(dict(prefix=prefix, path=str(source.resolve()), count=len(labels),
                manifest_sha256=manifest_hash, labels_sha256=labels_hash))
            for name, label in labels.items():
                target_name = prefix+'-'+name
                target = output/'images'/target_name
                shutil.copy2(source/'images'/name, target)
                if sha(target) != label['img_hash']:
                    raise ValueError(f'{source}: image checksum mismatch: {name}')
                combined[target_name] = label
                report['pages'].append(dict(image=target_name, source=prefix, source_image=name))
                if len(combined) % 100 == 0:
                    print(f'Copied and verified {len(combined)}/{total} images', flush=True)
                    write_json(output/'manifest.json', report)
        write_json(output/'labels.json', combined)
        report['status'] = 'complete'
        report['counts'] = {prefix:expected for prefix, _, expected, _ in specs}
        report['counts']['total'] = len(combined)
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(output/'manifest.json', report)
    print(f'Combined {len(combined)} pages: {output}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', type=Path, default=ROOT/'outputs/synth-jdoc-5000-v1')
    parser.add_argument('--tables', type=Path, default=ROOT/'outputs/synth-tables-2000-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/synth-mixed-7000-v1')
    parser.add_argument('--text-count', type=int, default=5000)
    parser.add_argument('--table-count', type=int, default=2000)
    parser.add_argument('--headings', type=Path)
    parser.add_argument('--heading-tables', type=Path)
    parser.add_argument('--heading-count', type=int, default=1000)
    parser.add_argument('--image-text', type=Path)
    parser.add_argument('--image-tables', type=Path)
    parser.add_argument('--image-text-count', type=int, default=1334)
    parser.add_argument('--image-table-count', type=int, default=666)
    args = parser.parse_args()
    if min(args.text_count,args.table_count,args.heading_count,args.image_text_count,args.image_table_count) < 1:
        parser.error('Counts must be positive')
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under textdet')
    combine(args.text,args.tables,args.output,args.text_count,args.table_count,
            args.headings,args.heading_tables,args.heading_count,
            args.image_text,args.image_tables,args.image_text_count,args.image_table_count)


if __name__ == '__main__':
    main()
