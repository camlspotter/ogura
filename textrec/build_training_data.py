"""Reproducible download -> charset -> direct selection -> substitution -> verification."""
from ogura.textrec.paths import CORPUS_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
from urllib.request import urlopen
from ogura.textrec.text_common import ROOT, export_plaintext
from ogura.textrec.diversity import VERSION

UNICODE_FILES={
 'Scripts-15.0.0.txt':('https://www.unicode.org/Public/15.0.0/ucd/Scripts.txt','cca85d830f46aece2e7c1459ef1249993dca8f2e46d51e869255be140d7ea4b0'),
 'ScriptExtensions-15.0.0.txt':('https://www.unicode.org/Public/15.0.0/ucd/ScriptExtensions.txt','7e07313d9d0bee42220c476b64485995130ae30917bbcf7780b602d677d7e33f'),
 'IVD_Sequences-2026-08-03.txt':('https://www.unicode.org/ivd/data/2026-08-03/IVD_Sequences.txt','2b466659c2bfde1c52e60bd9fc883ab14dfa5c583df024b80352420265d2cd87'),
}


def run(module,*args):
    cmd=[sys.executable,"-m","ogura.textrec."+module,*map(str,args)]
    print('Running: '+' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)


def complete(directory,goal,min_length,seed=None):
    if not (directory/'summary.json').exists():return False
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('diversity_version')!=VERSION or manifest.get('min_tokens',1)!=min_length or manifest['target_samples_per_entry']!=goal or (seed is not None and manifest.get('synthesis_seed')!=seed):
        raise ValueError(f'Configuration mismatch in {directory}; use a different --name')
    expected=hashlib.sha256((ROOT/'charset/selected/targets.jsonl').read_bytes()).hexdigest()
    if manifest['target_sha256']!=expected:raise ValueError(f'Character set mismatch in {directory}')
    for f in ('train.jsonl','targets.jsonl','coverage.jsonl','article_splits.jsonl'):
        if not (directory/f).exists():raise ValueError(f'Incomplete output: {directory/f}')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--goal',type=int,default=50)
    p.add_argument('--min-length',type=int,default=20)
    p.add_argument('--seed',type=int,default=20260915,help='Substitution RNG seed; article split remains fixed')
    p.add_argument('--name',help='New output suffix; default is the goal')
    p.add_argument('--targets-per-line', type=int, default=5)
    p.add_argument('--synthesis', choices=('hiragana', 'replacement'), default='hiragana')
    p.add_argument('--natural-base', type=Path, default=ROOT/'datasets/direct_100_len20_25_diverse', help='Reuse previously extracted and verified natural examples')
    p.add_argument('--fresh-natural', action='store_true', help='Extract natural examples from Wikipedia again')
    args=p.parse_args()
    if args.fresh_natural:args.natural_base=None
    elif not (args.natural_base/'train.jsonl').exists():p.error('Natural input missing; specify --natural-base or --fresh-natural')
    if args.goal<1:p.error('goal must be positive')
    if not 1<=args.min_length<=25:p.error('min-length must be 1..25')
    if not 1<=args.targets_per_line<=args.min_length:p.error('targets-per-line must be 1..min-length')
    name=args.name or f'{args.goal}_len{args.min_length}_25_{args.synthesis}' + (f'_mix{args.targets_per_line}' if args.synthesis == 'hiragana' else '')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):p.error('name must be letters, digits, underscore or hyphen')
    if not (ROOT/'cache/character_counts.jsonl').exists():run('analyze_wikipedia')
    folder=CORPUS_ROOT / 'unicode';folder.mkdir(parents=True,exist_ok=True)
    for filename,(url,sha) in UNICODE_FILES.items():
        path=folder/filename
        if not path.exists():
            with urlopen(url) as response: data=response.read()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError(f'Unexpected Unicode data: {url}')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError(f'Unicode checksum mismatch: {path}')
    if not (ROOT/'charset/candidates/candidates.jsonl').exists():run('classify_characters')
    if not (ROOT/'charset/selected/targets.jsonl').exists():run('refine_charset')
    natural=args.natural_base or ROOT/f'datasets/direct_{name}';nr=natural/'reports'
    final=ROOT/f'datasets/final_{name}';fr=final/'reports'
    if args.natural_base is None and not complete(natural,args.goal,args.min_length):run('select_wikipedia_direct','--goal',args.goal,'--min-length',args.min_length,'--output',natural,'--report-dir',nr)
    if not (nr/'verification.json').exists():run('verify_training_text','--output',natural,'--report-dir',nr)
    if args.synthesis == 'hiragana':
        if final.exists():
            meta=json.loads((final/'manifest.json').read_text())
            expected=hashlib.sha256((natural/'train.jsonl').read_bytes()).hexdigest()
            targets=hashlib.sha256((natural/'targets.jsonl').read_bytes()).hexdigest()
            if (meta.get('synthesis'),meta.get('target_samples_per_entry'),meta.get('seed'),meta.get('min_tokens'),meta.get('natural_sha256'),meta.get('target_sha256'),meta.get('targets_per_line',1)) != ('random_hiragana',args.goal,args.seed,args.min_length,expected,targets,args.targets_per_line):
                raise ValueError('Configuration mismatch; use a different --name')
            if not (fr/'verification.json').exists():raise ValueError('Incomplete generation; use a different --name')
            export_plaintext(final)
        else:
            run('generate_hiragana','--base',natural,'--output',final,'--goal',args.goal,'--seed',args.seed,'--min-length',args.min_length,'--targets-per-line',args.targets_per_line)
        print(f'Complete: {final / "train.txt"}',flush=True)
        return
    if not complete(final,args.goal,args.min_length,args.seed):run('synthesize_shortfalls','--base',natural,'--goal',args.goal,'--seed',args.seed,'--output',final,'--report-dir',fr)
    run('shuffle_training_text','--output',final,'--seed',args.seed)
    if not (fr/'verification.json').exists():run('verify_training_text','--output',final,'--report-dir',fr)
    exported = export_plaintext(final)
    print(f'Exported {exported:,} entries to {final / "train.txt"}', flush=True)
    for name in ('manifest.json','summary.json'):shutil.copyfile(final/name,fr/name)
    summary=json.loads((final/'summary.json').read_text())
    verification=json.loads((fr/'verification.json').read_text())
    (fr/'report.md').write_text(
        f"# 完成した学習テキスト\n\n合計 {summary['samples']:,}件。自然文 {summary['natural_samples']:,}件、置換合成 {summary['synthetic_samples']:,}件。\n\n"
        f"全 {summary['target_entries']:,}字種で目標 {args.goal}件以上。長さ{args.min_length}〜25文字、完全重複なし。\n\n"
        f"置換乱数シード {args.seed}。漢字は漢字、その他は同じUnicode一般カテゴリの文字に置換。\n"
        "元テキストは学習用Wikipediaの未使用区間からランダムに採る。重なる原文区間と共通する16文字断片を除外し、最後にシャッフルする。\n"
        "各合成行にbase_text・base_sample_id・replacementsを記録。元記事のstart/endは置換前の原文位置。\n\n"
        f"全件検証済み。原文照合 {verification['source_substrings_independently_verified']:,}件（合成全件を含む）。\n\n"
        f"置換元の再利用 {verification.get('reused_donor_texts', '未計測')}件。置換元が出力行にも存在するケース {verification.get('synthetic_donors_present_as_output', '未計測')}件。\n\n"
        "train.txt: 1行1件の統合データ。train.jsonl: 出典・置換履歴付き。synthetic.jsonl: 合成分のみ。\n",
        encoding='utf-8')
    print(f'Complete: {final / "train.txt"}',flush=True)


if __name__=='__main__':main()
