"""Select short, exact Wikipedia substrings with coverage-driven sampling."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata as ud

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[^\r\n][\ufe00-\ufe0f\U000e0100-\U000e01ef]?")
LINE = re.compile(r"[^\r\n\x85\u2028\u2029]+")
IVS = re.compile(r"(.)[\U000e0100-\U000e01ef]")


def strip_han_ivs(text):
    """Remove an ideographic selector only when attached to a Han character."""
    return IVS.sub(lambda m: m[1] if ud.name(m[1], "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")) else m[0], text)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokens(text):
    return [m.group() for m in TOKEN.finditer(text)]


def split_article(article_id, seed):
    bucket = int(digest(f"{seed}:{article_id}")[:16], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def emit(stream, row):
    text = json.dumps(row, ensure_ascii=False)
    for c in ("\x85", "\u2028", "\u2029"):
        text = text.replace(c, f"\\u{ord(c):04x}")
    stream.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--targets", type=Path, default=ROOT / "results/charset_refined/targets.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training_text")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/training_text")
    args = parser.parse_args()
    if args.goal < 1:
        parser.error("goal must be positive")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.jsonl").exists():
        parser.error("Output already exists; choose a fresh --output directory")
    target_file = args.targets
    target_bytes = target_file.read_bytes()
    targets = [json.loads(line) for line in target_bytes.decode().split("\n") if line]
    vocabulary = {r["character"] for r in targets}
    assert len(vocabulary) == len(targets) and vocabulary
    strip_ivs = not any(r.get("group") == "registered_ivs" for r in targets)
    (out / "targets.jsonl").write_bytes(target_bytes)
    frequencies = {r["character"]: r["ranking_occurrences"] for r in targets}
    counts, lengths, split_counts = Counter(), Counter(), Counter()
    anchor_articles = Counter()
    contexts = defaultdict(set)
    sample_hashes = set()
    reasons = Counter()
    rng = random.Random(args.seed)
    scanned = 0
    active_scalars = None
    length_weights = [1.0] * 25
    weights_at_sample = -1
    source_manifest = json.loads((ROOT / "results/source.json").read_text())
    metadata = {"seed": args.seed, "target_samples_per_entry": args.goal,
                "target_entries": len(targets), "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
                "source_repository": source_manifest["repository"], "source_revision": source_manifest["revision"],
                "source_config": source_manifest["config"],
                "max_tokens": 25, "length_sampling": "adaptive inverse-frequency weights on 1..25; three attempts per anchor; final attempt fits available run",
                "split": "SHA256(seed:article_id) first 64 bits modulo 100: train 0..89, validation 90..94, test 95..99",
                "context": "up to six tokens either side of anchor; unique per anchor character",
                "max_samples_per_anchor_per_article": 5,
                "tokenization": "one code point, except base plus one variation selector forms one token",
                "normalization": "none; exact source substrings; no line crossing; only target tokens",
                "limitations": "Greedy selection; failure to reach goal does not prove corpus exhaustion of all possible substrings."}
    metadata["strip_han_ivs"] = strip_ivs
    if strip_ivs:
        metadata["normalization"] = "strip supplementary ideographic selectors attached to Han; otherwise exact source substrings; offsets refer to original text"
    (out / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    with (out / "train.jsonl").open("w") as training, (out / "article_splits.jsonl").open("w") as splits:
        for file in source_manifest["files"]:
            path = ROOT / "data/20231101.ja" / Path(file["path"]).name
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["id", "url", "title", "text"]):
                for article in batch.to_pylist():
                    scanned += 1
                    split = split_article(article["id"], args.seed)
                    split_counts[split] += 1
                    emit(splits, {"article_id": article["id"], "split": split})
                    if split != "train":
                        continue
                    if active_scalars is None or scanned % 1000 == 0:
                        needy = {c for c in vocabulary if counts[c] < args.goal}
                        active_scalars = {c[0] for c in needy}
                    text = article["text"]
                    if active_scalars.isdisjoint(text):
                        continue
                    per_article = Counter()
                    contributed = set()
                    for line in LINE.finditer(text):
                        if active_scalars.isdisjoint(line.group()):
                            continue
                        matches = list(TOKEN.finditer(line.group()))
                        ts = [m.group() for m in matches]
                        if strip_ivs:
                            ts = [strip_han_ivs(c) for c in ts]
                        anchors = [i for i, c in enumerate(ts) if c in vocabulary and counts[c] < args.goal]
                        anchors.sort(key=lambda i: (frequencies[ts[i]], i))
                        occupied = []
                        # Invalid tokens are barriers; never silently delete them.
                        barriers = [-1] + [i for i, c in enumerate(ts) if c not in vocabulary] + [len(ts)]
                        for i in anchors:
                            char = ts[i]
                            if counts[char] >= args.goal or per_article[char] >= 5:
                                continue
                            boundary = bisect_left(barriers, i)
                            lo, hi = barriers[boundary-1]+1, barriers[boundary]
                            context = digest("".join(ts[max(lo,i-6):min(hi,i+7)]))
                            if context in contexts[char]:
                                reasons["repeated_anchor_context"] += 1
                                continue
                            for attempt in range(3):
                                if len(sample_hashes) // 1000 != weights_at_sample:
                                    weights_at_sample = len(sample_hashes) // 1000
                                    length_weights = [(lengths[n]+100)**-1.5 for n in range(1,26)]
                                    # Unique one-character strings have a finite ceiling.
                                    # Do not let that ceiling monopolize rejected proposals.
                                    length_weights[0] = min(length_weights[0], max(length_weights[1:]))
                                limit = min(25, hi-lo) if attempt == 2 else 25
                                length = rng.choices(range(1,limit+1), weights=length_weights[:limit])[0]
                                low_start, high_start = max(lo,i-length+1), min(i,hi-length)
                                if low_start > high_start:
                                    reasons["length_does_not_fit"] += 1
                                    continue
                                start = rng.randint(low_start,high_start)
                                end = start + length
                                if any(start < b and a < end for a,b in occupied):
                                    reasons["overlap_in_source_line"] += 1
                                    continue
                                if ts[start].isspace() or ts[end-1].isspace() or ud.category(ts[start][0]).startswith("M"):
                                    reasons["unsafe_edge"] += 1
                                    continue
                                if end < len(ts) and ud.category(ts[end][0]).startswith("M"):
                                    reasons["unsafe_edge"] += 1
                                    continue
                                sample = "".join(ts[start:end])
                                key = digest(sample)
                                if key in sample_hashes:
                                    reasons["duplicate_text"] += 1
                                    continue
                                begin = line.start() + matches[start].start()
                                finish = line.start() + matches[end-1].end()
                                source_sample = text[begin:finish]
                                assert (strip_han_ivs(source_sample) if strip_ivs else source_sample) == sample
                                sample_hashes.add(key)
                                contexts[char].add(context)
                                per_article[char] += 1
                                contributed.add(char)
                                occupied.append((start,end))
                                counts.update(set(ts[start:end]))
                                lengths[length] += 1
                                emit(training, {"sample_id": key, "text": sample, "length": length,
                                                "article_id": article["id"], "url": article["url"], "title": article["title"],
                                                "start": begin, "end": finish, "anchor": char})
                                break
                    anchor_articles.update(contributed)
                    if scanned % 10000 == 0:
                        print(f"{scanned:,} articles; {len(sample_hashes):,} samples; {sum(counts[c]>=args.goal for c in vocabulary):,} targets met", flush=True)
            print(f"Completed {path.name}: {scanned:,} articles; {len(sample_hashes):,} samples", flush=True)
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    status_counts = Counter()
    with (out / "coverage.jsonl").open("w") as coverage:
        for target in targets:
            c = target["character"]
            n = counts[c]
            status = "met" if n >= args.goal else "unobserved_in_corpus" if target["ranking_occurrences"] == 0 else "no_selected_sample" if n == 0 else "shortfall"
            status_counts[status] += 1
            emit(coverage, {"character": c, "codepoints": [f"U+{ord(x):04X}" for x in c],
                            "sample_count": n, "shortfall": max(0,args.goal-n), "status": status,
                            "corpus_occurrences": target["ranking_occurrences"],
                            "anchor_article_count": anchor_articles[c]})
    assert scanned == json.loads((ROOT / "results/summary.json").read_text())["articles"]
    summary = {**metadata, "articles": scanned, "article_splits": dict(split_counts),
               "samples": len(sample_hashes), "length_counts": dict(sorted(lengths.items())),
               "coverage_status": dict(status_counts), "rejected_attempts": dict(reasons)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for name in ("coverage.jsonl", "summary.json", "manifest.json"):
        (report_dir / name).write_bytes((out / name).read_bytes())
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
