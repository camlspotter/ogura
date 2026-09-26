"""Shared text normalization, provenance and deterministic article splitting."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata as ud

ROOT = Path(__file__).resolve().parents[2]
TOKEN = re.compile(r"[^\r\n][\ufe00-\ufe0f\U000e0100-\U000e01ef]?")
LINE = re.compile(r"[^\r\n\x85\u2028\u2029]+")
IVS = re.compile(r"(.)[\U000e0100-\U000e01ef]")


def strip_han_ivs(text):
    """Remove an ideographic selector only when attached to a Han character."""
    return IVS.sub(lambda m: m[1] if ud.name(m[1], "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")) else m[0], text)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_article(article_id, seed):
    bucket = int(digest(f"{seed}:{article_id}")[:16], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def emit(stream, row):
    text = json.dumps(row, ensure_ascii=False)
    for c in ("\x85", "\u2028", "\u2029"):
        text = text.replace(c, f"\\u{ord(c):04x}")
    stream.write(text + "\n")


def export_plaintext(directory):
    """Export JSONL in its current order as UTF-8, one entry per line."""
    temporary = directory / 'train.txt.tmp'
    count = 0
    try:
        with (directory / 'train.jsonl').open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8', newline='\n') as output:
            for line in source:
                text = json.loads(line)['text']
                if not text or any(c in text for c in '\r\n\x85\u2028\u2029'):
                    raise ValueError(f'Invalid single-line entry at row {count + 1}')
                output.write(text + '\n')
                count += 1
        temporary.replace(directory / 'train.txt')
    finally:
        temporary.unlink(missing_ok=True)
    return count
