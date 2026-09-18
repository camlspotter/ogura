"""Read-only alias candidate audit; never rewrites the vocabulary or training labels."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import unicodedata as ud
from PIL import ImageFont
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
SIZES = (28, 32, 40)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, default=ROOT/'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl')
    parser.add_argument('--confusables', type=Path, default=ROOT/'datasets/character_alias_audit/confusables.txt')
    parser.add_argument('--font-dir', type=Path, default=ROOT/'corpus/fonts')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/character_alias_audit')
    args = parser.parse_args()
    chars = [json.loads(s)['character'] for s in args.targets.read_text().splitlines()]
    targets = set(chars)
    files = ['NotoSansCJKjp-Regular.otf', 'NotoSansCJKjp-Bold.otf',
             'NotoSerifCJKjp-Regular.otf', 'NotoSerifCJKjp-Bold.otf',
             'NotoSansCJKjp-Light.otf', 'ZenMaruGothic-Light.ttf',
             'ZenMaruGothic-Regular.ttf', 'Tinos-Regular.ttf', 'Arimo[wght].ttf']
    paths = [args.font_dir/f for f in files]
    cmaps = {}
    for p in paths:
        with TTFont(p) as font:
            cmaps[p.name] = {cp:g for cp,g in font.getBestCmap().items() if g != '.notdef'}
    reasons = defaultdict(set)
    def add(group, reason):
        group = tuple(sorted(set(group) & targets, key=ord))
        if len(group) > 1:
            reasons[group].add(reason)
    def western(c):
        name = ud.name(c, '')
        return ((ord(c) <= 255 and name.startswith('LATIN')) or
                name.startswith('GREEK') or name.startswith('CYRILLIC')) and ud.category(c).startswith('L')
    mappings = {}
    for line in args.confusables.read_text().splitlines():
        parts = line.split('#',1)[0].strip().split(';')
        if len(parts) >= 2:
            source = ''.join(chr(int(cp,16)) for cp in parts[0].split())
            dest = ''.join(chr(int(cp,16)) for cp in parts[1].split())
            mappings[source] = dest
    western_groups = defaultdict(list)
    for c in chars:
        if western(c):
            skeleton = mappings.get(c,c)
            if len(skeleton) == 1:
                western_groups[skeleton].append(c)
    for group in western_groups.values():
        scripts = {ud.name(c).split()[0] for c in group}
        if len(scripts)>1:
            add(group, 'Unicode confusables: Latin/Greek/Cyrillic')
            if len(group) > 2:
                for first, second in itertools.combinations(group,2):
                    if ud.name(first).split()[0] != ud.name(second).split()[0]:
                        add([first,second], 'Unicode confusables subgroup')
    cjk_compat = []
    for c in chars:
        if ud.name(c,'').startswith('CJK COMPATIBILITY IDEOGRAPH'):
            normalized = ud.normalize('NFKC',c)
            cjk_compat.append(dict(character=c,normalized=normalized))
            if len(normalized)==1 and normalized!=c:
                add([c,normalized], 'CJK compatibility normalization')
    # Same cmap glyph among CJK characters, not all arbitrary look-alikes.
    for cmap in cmaps.values():
        reverse = defaultdict(list)
        for c in chars:
            if ud.name(c,'').startswith('CJK ') and ord(c) in cmap:
                reverse[cmap[ord(c)]].append(c)
        for group in reverse.values():
            if len(group)>1:
                for pair in itertools.combinations(group,2):
                    add(pair, 'CJK same cmap glyph')
    protected = [('0','O'), ('1','I','l'), ('口','ロ'), ('力','カ')]
    for group in protected:
        add(group,'Keep separate by current policy')
    for group in [('A','Ａ'), ('a','ａ'), ('0','０')]:
        add(group,'Width variants')
    @lru_cache(maxsize=None)
    def signature(c,name,size):
        font = ImageFont.truetype(str(args.font_dir/name),size,layout_engine=ImageFont.Layout.BASIC)
        mask = font.getmask(c, mode='L')
        return (font.getbbox(c,anchor='ls'),font.getlength(c),mask.size,hashlib.sha256(bytes(mask)).hexdigest())
    rows=[]
    for group, sources in sorted(reasons.items(),key=lambda item:tuple(map(ord,item[0]))):
        supported=[]; identical=[]; aliases=[]; partial=[]
        for name,cmap in cmaps.items():
            if not all(ord(c) in cmap for c in group):
                partial.append(name)
                continue
            supported.append(name)
            if len({cmap[ord(c)] for c in group})==1:aliases.append(name)
            if all(len({signature(c,name,size) for c in group})==1 for size in SIZES):
                identical.append(name)
        if 'Keep separate by current policy' in sources:
            decision='別クラス維持（方針）'
        elif not supported:
            decision='判定不能（共通の対応書体なし）'
        elif len(identical)==len(supported):
            decision='統合候補（比較できた全書体で一致）'
        elif identical:
            decision='統合検討（書体によって一致・不一致）'
        else:
            decision='要目視確認（完全一致なし）'
        rows.append(dict(characters=list(group),codepoints=[f'U+{ord(c):04X}' for c in group],
                         names=[ud.name(c,'') for c in group],sources=sorted(sources),
                         supported_fonts=supported,identical_fonts=identical,
                         same_glyph_fonts=aliases,unsupported_fonts=partial,decision=decision))
    report=dict(targets=len(chars),unicode_version=ud.unidata_version,sizes=SIZES,
                targets_sha256=hashlib.sha256(args.targets.read_bytes()).hexdigest(),
                confusables_sha256=hashlib.sha256(args.confusables.read_bytes()).hexdigest(),
                confusables_header=args.confusables.read_text().splitlines()[:10],
                fonts=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                compatibility_characters=cjk_compat,groups=rows)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 同形文字の比較（学習設定への反映は別管理）','',
           f'対象：現在の辞書{len(chars):,}字、9書体。28/32/40pxで単文字の画素・bbox・送り幅を完全比較。',
           '同一候補群とその部分ペアを含むので、行数は独立したクラス数ではない。',
           '今回の辞書には全角英字Ａ〜Ｚ・ａ〜ｚは未登録。全角数字は登録されている。',
           '共通して描画できない書体は比較から除外。完全一致は文脈中のカーニングや全解像度の一致を保証しない。',
           'Unicode confusablesは候補抽出用であり、同字形であることの証明ではない。',
           'NFKCはCJK互換漢字の候補抽出にのみ使用し、辞書・学習ラベルには適用していない。','',
           '| 文字 | コードポイント | 一致/比較可能書体 | 同一glyph書体数 | 暫定判断 |',
           '|---|---|---:|---:|---|']
    for r in rows:
        lines.append('| '+' / '.join(r['characters'])+' | '+' / '.join(r['codepoints'])+
                     f" | {len(r['identical_fonts'])}/{len(r['supported_fonts'])} | {len(r['same_glyph_fonts'])} | {r['decision']} |")
    (args.output/'candidates.md').write_text('\n'.join(lines)+'\n')
    print('groups',len(rows))
    for category in ('統合候補','統合検討','要目視確認','別クラス維持','判定不能'):
        subset=[r for r in rows if r['decision'].startswith(category)]
        print(category,len(subset))
        for r in subset:
            print(' '.join(r['characters']),f"{len(r['identical_fonts'])}/{len(r['supported_fonts'])}",
                  'same glyph',len(r['same_glyph_fonts']))


if __name__ == '__main__':
    main()
