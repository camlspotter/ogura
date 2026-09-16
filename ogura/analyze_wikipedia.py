"""Download a pinned Japanese Wikipedia snapshot and count raw code points."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata
from urllib.request import urlopen

import pyarrow.parquet as pq

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
REPO = "wikimedia/wikipedia"
CONFIG = "20231101.ja"
ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    raw = ROOT / "corpus/wikipedia" / CONFIG
    cache = ROOT / "cache/counts"
    output = ROOT / "cache"
    for directory in (raw, cache, output):
        directory.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{CONFIG}"
    with urlopen(api) as response:
        files = [item for item in json.load(response) if item["path"].endswith(".parquet")]
    assert len(files) == 15, "Unexpected snapshot inventory"
    write_json(raw.parent / "source.json", {
        "repository": REPO, "revision": REVISION, "config": CONFIG,
        "api": api, "files": files,
        "counting": "Raw Unicode code points in text only; no normalization or filtering",
        "unicode_version": unicodedata.unidata_version,
    })

    def download(item):
        path = raw / Path(item["path"]).name
        expected_hash = item.get("lfs", {}).get("oid")
        if not path.exists() or path.stat().st_size != item["size"]:
            temp = path.with_suffix(".part")
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}"
            subprocess.run(["curl", "-fL", "--silent", "--show-error", "--retry", "4",
                            "--output", str(temp), url], check=True)
            if temp.stat().st_size != item["size"]:
                raise ValueError(f"Size mismatch: {temp}")
            temp.replace(path)
        if expected_hash:
            with path.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"SHA256 mismatch: {path}")
        print(f"Ready: {path.name}", flush=True)
        return path

    total = Counter()
    article_counts = Counter()
    articles = 0
    empty = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path in pool.map(download, files):
            count_path = cache / (path.stem + ".json")
            if count_path.exists():
                result = json.loads(count_path.read_text())
            else:
                counts = Counter()
                docs = Counter()
                rows = 0
                empty_rows = 0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        if text is None:
                            raise ValueError("Unexpected null text")
                        counts.update(text)
                        docs.update(set(text))
                        rows += 1
                        empty_rows += not bool(text)
                result = {"counts": counts, "articles": docs, "rows": rows, "empty": empty_rows}
                write_json(count_path, result)
            total.update(result["counts"])
            article_counts.update(result["articles"])
            articles += result["rows"]
            empty += result["empty"]
            print(f"Counted: {path.name}; {articles:,} articles; {sum(total.values()):,} code points", flush=True)

    rows = []
    for char, count in sorted(total.items(), key=lambda pair: (-pair[1], ord(pair[0]))):
        rows.append({"character": char, "codepoint": f"U+{ord(char):04X}",
                     "name": unicodedata.name(char, "UNNAMED"),
                     "category": unicodedata.category(char), "occurrences": count,
                     "article_count": article_counts[char]})
    with (output / "character_counts.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    buckets = {"1-99": 0, "100-499": 0, "500+": 0}
    categories = {}
    for row in rows:
        count = row["occurrences"]
        buckets["1-99" if count < 100 else "100-499" if count < 500 else "500+"] += 1
        category = categories.setdefault(row["category"], {"types": 0, "occurrences": 0})
        category["types"] += 1
        category["occurrences"] += count
    summary = {"articles": articles, "empty_articles": empty,
               "codepoints": sum(total.values()), "distinct_codepoints": len(total),
               "occurrence_buckets": buckets, "unicode_categories": categories,
               "note": "Occurrences and article counts are not counts of distinct training samples."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
