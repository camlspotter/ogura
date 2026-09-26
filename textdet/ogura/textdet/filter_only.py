"""First-stage filtering only: never render pages or calculate bounding boxes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pymupdf as fitz

from .prepare import ROOT, filter_document, inspect_fonts, suspicious, write_json


def text_summary(page):
    raw = page.get_text('rawdict', sort=False,
                        flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES)
    chars = Counter()
    for block in raw['blocks']:
        if block['type'] == 0:
            for line in block['lines']:
                text = ''.join(c['c'] for span in line['spans'] for c in span['chars'])
                if text.strip():
                    chars.update(text)
    return {'page': page.number + 1,
            'char_count': sum(n for c, n in chars.items() if not c.isspace()),
            'suspicious_characters': [{'codepoint': f'U+{ord(c):04X}', 'count': n}
                                      for c, n in sorted(chars.items()) if suspicious(c)]}


def inspect_document(task):
    path, min_chars = task
    record = {'name': path.name, 'source': str(path), 'status': 'error'}
    try:
        stat = path.stat()
        record.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        fonts, warnings = inspect_fonts(path)
        record.update(fonts=fonts, font_warnings=warnings)
        missing = [f for f in fonts if not f['to_unicode']]
        # The whole PDF already fails: do not spend time extracting its pages.
        if not fonts or missing:
            record.update(status='skipped', text_checked=False,
                          reasons=[{'code': 'font_without_to_unicode', 'fonts': missing}]
                          if missing else [{'code': 'no_fonts'}])
            return record
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError('Password required')
            summaries = [text_summary(page) for page in pdf]
            reasons = filter_document(fonts, summaries, min_chars)
            record.update(status='skipped' if reasons else 'passed', text_checked=True,
                          pages=len(pdf), char_count=sum(p['char_count'] for p in summaries),
                          page_char_counts=[p['char_count'] for p in summaries], reasons=reasons)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during filtering')
    except Exception as error:
        record.update(status='error', error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / datetime.now().strftime('filter-%Y%m%d-%H%M%S'))
    parser.add_argument('--min-chars', type=int, default=100)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(ROOT) or args.min_chars < 1 or not 1 <= args.workers <= 8:
        parser.error('output must be under textdet; min-chars >= 1; workers 1..8')
    if shutil.which('pdffonts') is None:
        parser.error('pdffonts is required')
    paths = sorted(args.source.expanduser().resolve().glob('*.pdf'))
    if not paths:
        parser.error('No PDFs found')
    output.mkdir(parents=True, exist_ok=False)
    meta = {'schema_version': 1, 'mode': 'filter_only', 'status': 'running',
            'source': str(args.source.expanduser().resolve()), 'total': len(paths),
            'min_chars': args.min_chars,
            'font_policy': 'strict ToUnicode; font failures skip text extraction',
            'created_at': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'summary.json', meta)
    counts, reasons = Counter(), Counter()
    passed = []
    # Stream results to disk, without repeatedly rewriting a large manifest.
    with (output / 'documents.jsonl').open('w', encoding='utf-8') as stream, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, record in enumerate(pool.map(inspect_document, ((p, args.min_chars) for p in paths)), 1):
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            counts[record['status']] += 1
            reasons.update(r['code'] for r in record.get('reasons', []))
            if record['status'] == 'passed':
                passed.append({k: record[k] for k in ('name', 'source', 'pages', 'char_count', 'size_bytes', 'mtime_ns')})
            if i % 50 == 0 or i == len(paths):
                meta.update(processed=i, counts=dict(counts), reason_counts=dict(reasons))
                write_json(output / 'summary.json', meta)
                print(f'{i}/{len(paths)} {dict(counts)}', flush=True)
    write_json(output / 'passed.json', passed)
    meta.update(status='complete', passed_pages=sum(p['pages'] for p in passed))
    write_json(output / 'summary.json', meta)
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    print(output, flush=True)
    if counts['error']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
