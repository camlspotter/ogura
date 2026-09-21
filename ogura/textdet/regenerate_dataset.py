"""Pack a portable page recipe, then regenerate images from local PDFs (no network)."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import PIL
import pymupdf as fitz

from .prepare import ROOT, extract_page, render_page, write_json

SPLITS = ('train', 'val', 'test')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_hashes():
    return {name: sha256(ROOT/name) for name in ('prepare.py', 'glyph_bounds.py')}


def safe_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError(f'Expected a basename, got {name!r}')
    return name


def pack(experiment, pdf_root, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((experiment/'manifest.json').read_text())
    documents, pages = {}, []
    for split in SPLITS:
        labels = json.loads((experiment/split/'labels.json').read_text())
        for row in manifest['splits'][split]['items']:
            name = safe_name(row['source_pdf'])
            if name not in documents:
                path = pdf_root/name
                documents[name] = {'sha256': sha256(path)}
            label = labels[row['image']]
            pages.append({'source_pdf': name, 'page': row['page'], 'split': split,
                          'split_group': row['split_group'], 'image': row['image'],
                          'expected_boxes': len(label['polygons']),
                          'expected_dimensions': label['img_dimensions']})
    recipe = {'schema_version': 1, 'dpi': 150, 'seed': manifest['seed'],
              'selection_policy': 'Positive allowlist from exported experiment; never select additional PDF pages',
              'versions': {'pymupdf': fitz.VersionBind, 'pillow': PIL.__version__},
              'code_sha256': code_hashes(), 'experiment_manifest_sha256': sha256(experiment/'manifest.json'),
              'documents': documents, 'pages': pages,
              'quality_holdout_pages': [{'source_pdf': r['source_pdf'], 'page': r['page']}
                                        for r in manifest['experimental_quality_holdout']]}
    validate_recipe(recipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, recipe)
    return recipe


def validate_recipe(recipe):
    if recipe['schema_version'] != 1 or recipe['dpi'] != 150:
        raise ValueError('Unsupported recipe schema or resolution')
    seen, images, groups, documents = set(), set(), {}, {}
    for row in recipe['pages']:
        name = safe_name(row['source_pdf'])
        safe_name(row['image'])
        if row['split'] not in SPLITS or type(row['page']) is not int or row['page'] < 1:
            raise ValueError('Invalid split or page')
        if row['image'] != f'{Path(name).stem}_page-{row["page"]:04d}_page.png':
            raise ValueError('Unexpected image name')
        key = (name, row['page'])
        if key in seen or row['image'] in images:
            raise ValueError('Duplicate recipe page or image')
        seen.add(key); images.add(row['image'])
        for mapping, group in ((groups, row['split_group']), (documents, name)):
            if mapping.setdefault(group, row['split']) != row['split']:
                raise ValueError('Document or similarity group crosses splits')
        if name not in recipe['documents']:
            raise ValueError('Missing PDF digest')
    if not seen:
        raise ValueError('Empty recipe')


def generate_document(task):
    source, rows, output, dpi = task
    result = []
    with fitz.open(source) as pdf:
        if pdf.needs_pass:
            raise ValueError(f'Password required: {source}')
        for row in rows:
            page = pdf[row['page']-1]
            data = extract_page(page)
            if len(data['lines']) != row['expected_boxes']:
                raise ValueError(f'BBox count differs: {source.name} page {row["page"]}')
            data.update(source=str(source), source_pdf=source.name)
            directory = output/row['split']
            prefix = row['image'].removesuffix('_page.png')
            render_page(page, data, directory/'review', dpi, prefix=prefix)
            if [data['image_height'], data['image_width']] != row['expected_dimensions']:
                raise ValueError('Rendered dimensions differ from recipe')
            image = directory/'images'/row['image']
            (directory/'review'/row['image']).replace(image)
            data['image'] = '../images/'+row['image']
            write_json(directory/'review'/f'{prefix}_labels.json', data)
            polygons = []
            for line in data['lines']:
                x0,y0,x1,y1 = line['bbox_pixels']
                x0,y0,x1,y1 = max(0,x0),max(0,y0),min(data['image_width'],x1),min(data['image_height'],y1)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError('Invalid clipped box')
                polygons.append([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            label = {'img_dimensions':row['expected_dimensions'], 'img_hash':sha256(image),'polygons':polygons}
            result.append((row, label))
    return result


def generate(recipe_path, pdf_root, output, workers=2, limit=None):
    recipe = json.loads(recipe_path.read_text())
    validate_recipe(recipe)
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= workers <= 8 or (limit is not None and limit < 1):
        raise ValueError('workers must be 1..8 and limit must be positive')
    if recipe['code_sha256'] != code_hashes():
        raise ValueError('Extraction code differs from recipe; use the matching code or pack again')
    if recipe['versions'] != {'pymupdf':fitz.VersionBind,'pillow':PIL.__version__}:
        raise ValueError('Install the exact versions in textdet/requirements.txt')
    pages = recipe['pages'][:limit] if limit else recipe['pages']
    grouped = {}
    for row in pages:
        grouped.setdefault(row['source_pdf'], []).append(row)
    # Validate every selected source before creating any images.
    for name in grouped:
        if sha256(pdf_root/name) != recipe['documents'][name]['sha256']:
            raise ValueError(f'PDF contents differ: {name}')
    output.mkdir(parents=True)
    for split in SPLITS:
        (output/split/'images').mkdir(parents=True)
        (output/split/'review').mkdir()
    manifest = {'status':'running','recipe_sha256':sha256(recipe_path), 'pdf_root':str(pdf_root.resolve()),
                'versions':recipe['versions'], 'code_sha256':recipe['code_sha256'],
                'partial_run':len(pages) != len(recipe['pages']), 'pages':[]}
    write_json(output/'manifest.json',manifest)
    labels = {split:{} for split in SPLITS}
    try:
        tasks = [(pdf_root/name,rows,output,recipe['dpi']) for name,rows in grouped.items()]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in pool.map(generate_document,tasks):
                for row,label in batch:
                    labels[row['split']][row['image']] = label
                    manifest['pages'].append(row)
                print(f'{len(manifest["pages"])}/{len(pages)} pages',flush=True)
        for split in SPLITS:
            write_json(output/split/'labels.json',labels[split])
        manifest.update(status='complete',split_counts=dict(Counter(p['split'] for p in pages)))
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    packing=commands.add_parser('pack',help='Run on the machine holding the reviewed experiment')
    packing.add_argument('--experiment',type=Path,default=ROOT/'outputs/experiment-v1')
    packing.add_argument('--pdf-root',type=Path,required=True)
    packing.add_argument('--output',type=Path,default=ROOT/'outputs/experiment-v1-recipe.json')
    rendering=commands.add_parser('generate',help='Run locally on the GPU machine; GPU not needed')
    rendering.add_argument('--recipe',type=Path,required=True)
    rendering.add_argument('--pdf-root',type=Path,required=True)
    rendering.add_argument('--output',type=Path,default=ROOT/'outputs/experiment-v1-regenerated')
    rendering.add_argument('--workers',type=int,default=2)
    rendering.add_argument('--limit',type=int,help='Smoke test only: first N recipe pages')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error('Output must be under ogura/textdet')
    if args.command=='pack':
        recipe=pack(args.experiment,args.pdf_root,args.output)
        print(f'{len(recipe["pages"])} pages; {len(recipe["documents"])} PDFs; {args.output}')
    else:
        generate(args.recipe,args.pdf_root,args.output,args.workers,args.limit)

if __name__=='__main__':main()
