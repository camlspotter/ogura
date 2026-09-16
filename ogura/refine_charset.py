"""Narrow the initial character set to the approved modern script ranges."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata as ud

from ogura.classify_characters import dump_lines

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "charset/selected"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    initial = [json.loads(line) for line in (ROOT / "charset/candidates/candidates.jsonl").open()]
    observed = {r["character"]: r for r in map(json.loads, (ROOT / "cache/character_counts.jsonl").open())}
    latin = {chr(cp) for cp in range(256) if "LATIN" in ud.name(chr(cp), "") and ud.category(chr(cp)).startswith("L")} | set("ªº")
    greek = set("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψως")
    cyrillic = {chr(cp) for cp in range(0x410,0x450)} | set("Ёё")
    kana = {chr(cp) for lo,hi in [(0x3041,0x3096),(0x3099,0x309F),(0x30A0,0x30FF),(0x31F0,0x31FF),(0xFF61,0xFF9F)]
            for cp in range(lo,hi+1) if ud.category(chr(cp)) != "Cn"}
    required = latin | greek | cyrillic | kana | {chr(cp) for cp in range(0x20,0x7F)} | {"\u3000"}
    rows, removed = {}, []
    for r in initial:
        char = r["character"]
        groups = set(r.get("coverage_groups",[]))
        name = r.get("name", "")
        discard = None
        protected_punctuation = r.get("group") in {"ascii", "space", "cjk_punctuation_and_symbols", "japanese_iteration_and_marks"}
        if r.get("group") == "registered_ivs":
            discard = "ivs_collapsed_to_base"
        elif protected_punctuation:
            pass
        elif ("latin" in groups or "LATIN" in name) and char not in latin:
            discard = "outside_iso8859_1_latin"
        elif "greek" in groups and char not in greek:
            discard = "outside_basic_greek"
        elif ("kana" in groups or r["group"] == "kana") and char not in kana:
            discard = "outside_modern_kana_ranges"
        if discard:
            removed.append(dict(r, removal_reason=discard))
        else:
            rows[char] = dict(r)
    for char in required:
        if char not in rows:
            base = observed.get(char, {"character": char, "codepoint": f"U+{ord(char):04X}",
                "name": ud.name(char,"UNNAMED"), "category": ud.category(char), "occurrences":0,"article_count":0})
            rows[char] = dict(base, standalone_occurrences=base["occurrences"], ranking_occurrences=base["occurrences"], status="candidate")
    for char,r in rows.items():
        if char in latin: group="latin_iso8859_1"
        elif char in greek: group="greek_basic"
        elif char in cyrillic: group="cyrillic_russian_basic"
        elif char in kana: group="modern_kana_and_marks"
        elif r.get("group")=="han": group="han"
        else: group="other_digits_symbols_spaces"
        r["group"] = group
        r["required"] = char in required
        r.pop("coverage_groups",None)
    ordered=sorted(rows.values(),key=lambda r:(-r["ranking_occurrences"],r["character"]))
    for rank,r in enumerate(ordered,1):r["frequency_rank"]=rank
    dump_lines(OUT/"targets.jsonl",ordered)
    dump_lines(OUT/"removed.jsonl",removed)
    initial_chars = {x["character"] for x in initial}
    added=[r for r in ordered if r["character"] not in initial_chars]
    dump_lines(OUT/"added.jsonl",added)
    summary={"entries":len(rows),"previous_entries":len(initial),"removed":len(removed),"added":len(added),
             "groups":dict(Counter(r["group"] for r in ordered)),
             "unobserved":dict(Counter(r["group"] for r in ordered if r["ranking_occurrences"]==0)),
             "target_sha256":hashlib.sha256((OUT/"targets.jsonl").read_bytes()).hexdigest(),
             "policy":{"latin":"ISO-8859-1 encoded Latin letters only; excludes fullwidth Latin letters",
                       "greek":"24 uppercase + 24 lowercase + final sigma; no accents or symbol variants",
                       "cyrillic":"Russian basic 33 uppercase + 33 lowercase, including Yo",
                       "kana":"3041-3096, 3099-309F, 30A0-30FF, 31F0-31FF, FF61-FF9F; assigned code points",
                       "han_ivs":"Han inventory unchanged; IVS entries removed and selectors stripped during extraction"}}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    assert len(greek)==49 and len(cyrillic)==66
    assert all("HENTAIGANA" not in r.get("name","") and "MINNAN" not in r.get("name","") for r in ordered)
    assert all(ord(c)<=255 for c in latin)
    assert set(rows)==({r["character"] for r in initial}-{r["character"] for r in removed}) | {r["character"] for r in added}
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
