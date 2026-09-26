"""Render two representative pages per previously filtered PDF."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path

import pymupdf as fitz

from .prepare import ROOT, extract_page, render_page, write_json


def choose_pages(page_char_counts):
    eligible = [number for number, count in enumerate(page_char_counts, 1) if count > 0]
    count = len(eligible)
    if not count:
        return []
    return sorted({eligible[(count + 2) // 3 - 1], eligible[(2 * count + 2) // 3 - 1]})


def load_records(input_list, filter_records):
    records = json.loads(input_list.read_text())
    by_name = {record['name']: record for record in records}
    if len(by_name) != len(records):
        raise ValueError('Duplicate PDF names')
    with filter_records.open() as stream:
        for line in stream:
            saved = json.loads(line)
            record = by_name.get(saved['name'])
            if record is None:
                continue
            if saved['status'] != 'passed' or any(
                saved[key] != record[key] for key in ('source', 'pages', 'char_count', 'size_bytes', 'mtime_ns')
            ):
                raise ValueError(f'Filter record mismatch: {record["name"]}')
            counts = saved['page_char_counts']
            if len(counts) != record['pages'] or sum(counts) != record['char_count']:
                raise ValueError(f'Invalid page counts: {record["name"]}')
            record['page_char_counts'] = counts
    for record in records:
        if 'page_char_counts' not in record:
            raise ValueError(f'Missing page counts: {record["name"]}')
    return records


def process(task):
    record, output, dpi = task
    path = Path(record['source'])
    result = {'name': record['name'], 'source': str(path), 'pages': [], 'status': 'error'}
    try:
        stat = path.stat()
        if stat.st_size != record['size_bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('PDF changed since first-stage filtering')
        with fitz.open(path) as pdf:
            if len(pdf) != record['pages'] or pdf.needs_pass:
                raise ValueError('Page count changed or password required')
            for number in choose_pages(record['page_char_counts']):
                prefix = f'{path.stem}_page-{number:04d}'
                data = extract_page(pdf[number - 1])
                if not data['char_count']:
                    raise ValueError(f'Selected page {number} has no extracted characters')
                data.update(source=str(path), source_pdf=path.name)
                render_page(pdf[number - 1], data, output, dpi, prefix=prefix)
                result['pages'].append({'page': number, 'bbox_image': f'{prefix}_bbox.png',
                                        'image': f'{prefix}_page.png', 'labels': f'{prefix}_labels.json',
                                        'lines': len(data['lines'])})
        result['status'] = 'complete'
    except Exception as error:
        result['error'] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', type=Path, required=True, dest='input_list')
    parser.add_argument('--filter-records', type=Path, help='Defaults to documents.jsonl beside --list')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / datetime.now().strftime('samples-%Y%m%d-%H%M%S'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or not 1 <= args.workers <= 8:
        parser.error('Output must be under textdet; workers must be 1..8')
    filter_records = args.filter_records or args.input_list.with_name('documents.jsonl')
    records = load_records(args.input_list, filter_records)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'input_list': str(args.input_list.resolve()),
                'filter_records': str(filter_records.resolve()),
                'page_selection': 'Exclude zero-character pages; select ceil(N/3), ceil(2*N/3) among remaining N pages, deduplicated; one-based original PDF page indices',
                'dpi': 150, 'documents': []}
    write_json(output / 'manifest.json', manifest)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(process, ((r, output, 150) for r in records)), 1):
            manifest['documents'].append(result)
            write_json(output / 'manifest.json', manifest)
            if i % 20 == 0 or result['status'] == 'error':
                print(f'{i}/{len(records)} {result["name"]} {result["status"]}', flush=True)
    manifest.update(status='complete', document_count=len(records),
                    image_count=sum(len(r['pages']) for r in manifest['documents']),
                    error_count=sum(r['status'] == 'error' for r in manifest['documents']))
    write_json(output / 'manifest.json', manifest)
    print({k: manifest[k] for k in ('document_count', 'image_count', 'error_count')}, flush=True)
    if manifest['error_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
