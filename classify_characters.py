"""Classify observed code points and count registered IVS in Wikipedia text."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata as ud

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "charset"
IVD = ROOT / "data/charset_sources/IVD_Sequences-2026-08-03.txt"
SCRIPTS = ROOT / "data/charset_sources/Scripts-15.0.0.txt"
SCRIPT_EXTENSIONS = ROOT / "data/charset_sources/ScriptExtensions-15.0.0.txt"
VS = re.compile("[\U000e0100-\U000e01ef]")


def required_characters():
    """Finite coverage by Unicode Script/Script_Extensions, independent of corpus."""
    groups = defaultdict(set)
    aliases = {"Hiragana": "kana", "Katakana": "kana", "Latin": "latin", "Greek": "greek",
               "Hira": "kana", "Kana": "kana", "Latn": "latin", "Grek": "greek"}
    for path in (SCRIPTS, SCRIPT_EXTENSIONS):
        for line in path.read_text().splitlines():
            body = line.split("#", 1)[0].strip()
            if not body:
                continue
            span, scripts = map(str.strip, body.split(";"))
            matched = {aliases[s] for s in scripts.split() if s in aliases}
            if not matched:
                continue
            bounds = span.split("..")
            for cp in range(int(bounds[0], 16), int(bounds[-1], 16) + 1):
                groups[chr(cp)].update(matched)
    # Always retain ASCII printable characters and fullwidth digits/letters/symbols.
    for cp in list(range(0x20, 0x7F)) + list(range(0xFF01, 0xFF61)) + list(range(0xFFE0, 0xFFE7)) + [0x3000]:
        groups[chr(cp)].add("basic_alphanumeric_symbols")
    return groups


def dump_lines(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            # Escape Unicode line separators as well as JSON control characters.
            line = json.dumps(row, ensure_ascii=False)
            for c in ("\x85", "\u2028", "\u2029"):
                line = line.replace(c, f"\\u{ord(c):04x}")
            stream.write(line + "\n")


def classify(c):
    n, cat, cp = ud.name(c, ""), ud.category(c), ord(c)
    if 0xE0100 <= cp <= 0xE01EF or 0xFE00 <= cp <= 0xFE0F:
        return "variation_selector", "exclude_standalone"
    if cat[0] == "C" or cat in ("Zl", "Zp"):
        return "control_private_or_unassigned", "review" if cat in ("Co", "Cn") else "exclude"
    if c in (" ", "\u3000"):
        return "space", "candidate"
    if cat == "Zs":
        return "other_space", "review"
    if 0x21 <= cp <= 0x7E:
        return "ascii", "candidate"
    if n.startswith(("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")):
        return "han", "candidate"
    if "HIRAGANA" in n or "KATAKANA" in n:
        return "kana", "candidate"
    if c in "々〆〇〻〼":
        return "japanese_iteration_and_marks", "candidate"
    if 0xFF01 <= cp <= 0xFF60 or 0xFFE0 <= cp <= 0xFFE6:
        return "fullwidth_alphanumeric_and_symbols", "candidate"
    if 0x3000 <= cp <= 0x303F and cat[0] in "PS":
        return "cjk_punctuation_and_symbols", "candidate"
    if cat[0] in "PS":
        return "other_punctuation_and_symbols", "review"
    if "LATIN" in n:
        return "extended_latin", "review"
    if cat[0] == "M":
        return "combining_mark", "review"
    return "other_script_or_number", "review"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/character_counts.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row["group"], row["status"] = classify(row["character"])
    required = required_characters()
    observed = {row["character"] for row in rows}
    required_rows = []
    for row in rows:
        row["required"] = row["character"] in required
        if row["required"]:
            row["status"] = "candidate"
            row["coverage_groups"] = sorted(required[row["character"]])
            required_rows.append(row)
    missing_rows = []
    for char in sorted(required.keys() - observed):
        row = {"character": char, "codepoint": f"U+{ord(char):04X}",
               "name": ud.name(char, "UNNAMED"), "category": ud.category(char),
               "occurrences": 0, "article_count": 0, "standalone_occurrences": 0,
               "group": "required_unobserved", "status": "candidate", "required": True,
               "coverage_groups": sorted(required[char])}
        missing_rows.append(row)
        required_rows.append(row)
    registry = defaultdict(list)
    for line in IVD.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sequence, collection, identifier = map(str.strip, line.split(";")[:3])
        sequence = "".join(chr(int(cp, 16)) for cp in sequence.split())
        registry[sequence].append({"collection": collection, "identifier": identifier})
    counts, docs, invalid = Counter(), Counter(), Counter()
    scanned = 0
    for path in sorted((ROOT / "data/20231101.ja").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
            for text in batch.column(0).to_pylist():
                seen = set()
                for match in VS.finditer(text):
                    sequence = text[max(0, match.start()-1):match.end()]
                    if sequence in registry:
                        counts[sequence] += 1
                        seen.add(sequence)
                    else:
                        invalid[sequence] += 1
                docs.update(seen)
                scanned += 1
        print(f"IVS scanned: {scanned:,} articles", flush=True)
    ivs_rows = []
    for sequence, count in sorted(counts.items(), key=lambda p: (-p[1], p[0])):
        ivs_rows.append({"character": sequence, "codepoints": [f"U+{ord(c):04X}" for c in sequence],
                         "occurrences": count, "article_count": docs[sequence],
                         "registrations": registry[sequence], "group": "registered_ivs", "status": "candidate"})
    # Single-code-point totals include occurrences followed by registered IVS.
    # Keep the raw total, but remove those occurrences from standalone ranking.
    followed = Counter()
    for sequence, count in counts.items():
        followed[sequence[0]] += count
    for row in rows:
        row["standalone_occurrences"] = row["occurrences"] - followed[row["character"]]
        assert row["standalone_occurrences"] >= 0
    candidates = [dict(row, ranking_occurrences=row["standalone_occurrences"])
                  for row in rows if row["status"] == "candidate"]
    candidates.extend(dict(row, ranking_occurrences=0) for row in missing_rows)
    candidates.extend(dict(row, ranking_occurrences=row["occurrences"]) for row in ivs_rows)
    candidates.sort(key=lambda row: (-row["ranking_occurrences"], row["character"]))
    for rank, row in enumerate(candidates, 1):
        row["frequency_rank"] = rank
    dump_lines(OUT / "classified_codepoints.jsonl", rows)
    dump_lines(OUT / "candidates.jsonl", candidates)
    dump_lines(OUT / "required_characters.jsonl", sorted(required_rows, key=lambda row: row["character"]))
    dump_lines(OUT / "required_unobserved.jsonl", missing_rows)
    dump_lines(OUT / "review.jsonl", [row for row in rows if row["status"] == "review"])
    dump_lines(OUT / "excluded.jsonl", [row for row in rows if row["status"].startswith("exclude")])
    dump_lines(OUT / "ivs.jsonl", ivs_rows)
    dump_lines(OUT / "unregistered_vs_sequences.jsonl", [
        {"sequence": seq, "codepoints": [f"U+{ord(c):04X}" for c in seq], "occurrences": count}
        for seq, count in invalid.most_common()])
    groups = Counter(row["group"] for row in rows)
    summary = {"articles_scanned": scanned, "unicode_version": ud.unidata_version,
               "classified_codepoints": len(rows), "groups": dict(groups),
               "status_counts": dict(Counter(row["status"] for row in rows)),
               "candidate_entries_including_ivs": len(candidates),
               "required_entries": len(required), "required_unobserved": len(missing_rows),
               "required_coverage_groups": dict(Counter(g for groups in required.values() for g in groups)),
               "required_sources": [
                   {"url": f"https://www.unicode.org/Public/15.0.0/ucd/{name}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in (("Scripts.txt", SCRIPTS), ("ScriptExtensions.txt", SCRIPT_EXTENSIONS))],
               "registered_ivs_types": len(counts), "registered_ivs_occurrences": sum(counts.values()),
               "unregistered_vs_sequence_types": len(invalid),
               "unregistered_vs_occurrences": sum(invalid.values()),
               "ivd_url": "https://www.unicode.org/ivd/data/2026-08-03/IVD_Sequences.txt",
               "ivd_sha256": hashlib.sha256(IVD.read_bytes()).hexdigest(),
               "candidate_frequency_buckets": dict(Counter(
                   "500+" if r["ranking_occurrences"] >= 500 else "100-499" if r["ranking_occurrences"] >= 100 else "0-99"
                   for r in candidates)),
               "note": "Provisional candidates, not a finalized dictionary. Han characters are not classified by language."}
    assert sum(counts.values()) + sum(invalid.values()) == sum(
        r["occurrences"] for r in rows if 0xE0100 <= ord(r["character"]) <= 0xE01EF)
    expected = json.loads((ROOT / "results/summary.json").read_text())["articles"]
    assert scanned == expected
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
