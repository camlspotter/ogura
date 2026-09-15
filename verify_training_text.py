"""Verify all sample invariants and source text for a deterministic sample."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from extract_training_text import digest, split_article, strip_han_ivs
from synthesize_shortfalls import apply_replacements

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/training_text")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/training_text")
    args = parser.parse_args()
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    targets = {json.loads(line)["character"] for line in (out / "targets.jsonl").open()}
    hashes, counts, lengths = set(), Counter(), Counter()
    source_checks = defaultdict(list)
    samples = 0
    previous_article = None
    intervals = []
    per_article_anchors = Counter()

    def check_article():
        ordered = sorted(intervals)
        if not manifest.get("allow_source_overlap", False):
            assert all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:]))
        cap = manifest.get("max_samples_per_anchor_per_article", 5)
        if cap is not None:
            assert all(n <= cap for n in per_article_anchors.values())

    plaintext = (out / "train.txt.tmp").open("w")
    for line in (out / "train.jsonl").open():
        row = json.loads(line)
        text = row["text"]
        if row.get("synthetic", False):
            assert digest(row["base_text"]) == row["base_sample_id"]
            assert apply_replacements(row["base_text"], row["replacements"]) == text
        if row["article_id"] != previous_article:
            check_article()
            intervals.clear()
            per_article_anchors.clear()
            previous_article = row["article_id"]
        intervals.append((row["start"],row["end"]))
        per_article_anchors[row["anchor"]] += 1
        # Independent scanner: variation selectors attach to the preceding scalar.
        ts = []
        for char in text:
            cp = ord(char)
            if 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF:
                assert ts
                ts[-1] += char
            else:
                ts.append(char)
        assert manifest.get("min_tokens", 1) <= len(ts) <= 25 and row["length"] == len(ts)
        assert set(ts) <= targets and row["anchor"] in ts
        assert re.search(r"[\r\n\x85\u2028\u2029]", text) is None
        assert row["sample_id"] == digest(text) and row["sample_id"] not in hashes
        assert split_article(row["article_id"],manifest["seed"]) == "train"
        if manifest.get("strip_han_ivs", False):
            assert len(text) <= row["end"] - row["start"] <= 2 * len(text)
            assert strip_han_ivs(text) == text
        else:
            assert len(text) == row["end"] - row["start"]
        hashes.add(row["sample_id"])
        counts.update(set(ts))
        lengths[len(ts)] += 1
        samples += 1
        plaintext.write(text + "\n")
        if row.get("synthetic", False) or int(row["sample_id"][:8],16) % 1000 == 0 or len(text) != row["end"] - row["start"]:
            source_checks[row["article_id"]].append(row)
    check_article()
    assert samples == summary["samples"]
    assert dict(lengths) == {int(k):v for k,v in summary["length_counts"].items()}
    for line in (out / "coverage.jsonl").open():
        row = json.loads(line)
        assert counts[row["character"]] == row["sample_count"]
    split_counts = Counter()
    seen_articles = set()
    for line in (out / "article_splits.jsonl").open():
        row = json.loads(line)
        assert row["article_id"] not in seen_articles
        seen_articles.add(row["article_id"])
        assert row["split"] == split_article(row["article_id"],manifest["seed"])
        split_counts[row["split"]] += 1
    assert dict(split_counts) == summary["article_splits"]
    assert len(seen_articles) == summary["articles"]
    source_verified = 0
    for path in sorted((ROOT / "data/20231101.ja").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=["id","text"]):
            for article in batch.to_pylist():
                if article["id"] not in source_checks:
                    continue
                for row in source_checks.pop(article["id"]):
                    source_text = article["text"][row["start"]:row["end"]]
                    if manifest.get("strip_han_ivs", False):
                        source_text = strip_han_ivs(source_text)
                    if row.get("synthetic", False):
                        assert source_text == row["base_text"]
                        source_text = apply_replacements(source_text, row["replacements"])
                    assert source_text == row["text"]
                    source_verified += 1
    assert not source_checks
    plaintext.close()
    (out / "train.txt.tmp").replace(out / "train.txt")
    report = {"all_samples_verified": samples, "unique_texts": len(hashes),
              "source_substrings_independently_verified": source_verified,
              "source_check_selection": "sample SHA256 first 32 bits modulo 1000 equals zero, plus all synthetic and IVS-normalized samples",
              "checks": [f"{manifest.get('min_tokens', 1)}..25 tokens", "atomic variation sequences", "target vocabulary",
                         "no line crossing", "unique text hashes", "train-only article split",
                         "source overlap and anchor cap according to manifest", "complete article split manifest",
                         "exact coverage and length counts", "sampled source equality", "all synthetic replacements and source equality"]}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    result = args.report_dir / "verification.json"
    result.write_text(json.dumps(report,indent=2) + "\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
