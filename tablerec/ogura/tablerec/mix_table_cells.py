"""Combine equal-sized JSON datasets without changing recipes or storing images."""
import argparse
import hashlib
import json
from pathlib import Path

from .synth_table_cells import TableCellDataset


def mix(inputs, output):
    if len(inputs) < 2:
        raise ValueError('At least two inputs are required')
    datasets = [TableCellDataset(path) for path in inputs]
    if len({len(dataset) for dataset in datasets}) != 1:
        raise ValueError('Input counts must match for an equal mixture')
    manifests = [json.loads((dataset.root/'manifest.json').read_text()) for dataset in datasets]
    if len({manifest['split'] for manifest in manifests}) != 1:
        raise ValueError('Input splits differ')
    rows, seen = [], set()
    for index,dataset in enumerate(datasets):
        for record in dataset.records:
            path = (dataset.root/record['recipe']).resolve()
            if not path.is_relative_to(dataset.root.resolve()):
                raise ValueError('Recipe path escapes dataset')
            content = path.read_bytes()
            recipe = json.loads(content)
            if recipe['seed'] in seen:
                raise ValueError('Duplicate recipe seeds across inputs')
            seen.add(recipe['seed'])
            name = f'source-{index}-{record["id"]}'
            if Path(name).name != name:
                raise ValueError('Invalid sample id')
            rows.append((dict(record,id=name,recipe=f'recipes/{name}.json',source_dataset=index),content))
    output.mkdir(parents=True,exist_ok=False)
    (output/'recipes').mkdir()
    metadata = dict(schema_version=max(m['schema_version'] for m in manifests),status='generating',
                    storage='json-only',count=len(rows),split=manifests[0]['split'],
                    sources=[dict(path=str(Path(path).resolve()),count=len(dataset),
                                  manifest_sha256=hashlib.sha256((dataset.root/'manifest.json').read_bytes()).hexdigest())
                             for path,dataset in zip(inputs,datasets)])
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with (output/'samples.jsonl').open('w') as records:
        for record,content in rows:
            (output/record['recipe']).write_bytes(content)
            records.write(json.dumps(record)+'\n')
    metadata['status']='complete'
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(mix(args.input,args.output)))
    except (ValueError,OSError) as exc:
        parser.exit(1,f'{exc}\n')


if __name__=='__main__':
    main()
