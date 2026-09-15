"""Reproducible download -> charset -> direct selection -> substitution -> verification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
from extract_training_text import ROOT

UNICODE_FILES={
 'Scripts-15.0.0.txt':('https://www.unicode.org/Public/15.0.0/ucd/Scripts.txt','cca85d830f46aece2e7c1459ef1249993dca8f2e46d51e869255be140d7ea4b0'),
 'ScriptExtensions-15.0.0.txt':('https://www.unicode.org/Public/15.0.0/ucd/ScriptExtensions.txt','7e07313d9d0bee42220c476b64485995130ae30917bbcf7780b602d677d7e33f'),
 'IVD_Sequences-2026-08-03.txt':('https://www.unicode.org/ivd/data/2026-08-03/IVD_Sequences.txt','2b466659c2bfde1c52e60bd9fc883ab14dfa5c583df024b80352420265d2cd87'),
}


def run(script,*args):
    cmd=[sys.executable,str(ROOT/script),*map(str,args)]
    print('Running: '+' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)


def complete(directory,goal,min_length,seed=None):
    if not (directory/'summary.json').exists():return False
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('min_tokens',1)!=min_length or manifest['target_samples_per_entry']!=goal or (seed is not None and manifest.get('synthesis_seed')!=seed):
        raise ValueError(f'Configuration mismatch in {directory}; use a different --name')
    expected=hashlib.sha256((ROOT/'results/charset_refined/targets.jsonl').read_bytes()).hexdigest()
    if manifest['target_sha256']!=expected:raise ValueError(f'Character set mismatch in {directory}')
    for f in ('train.jsonl','targets.jsonl','coverage.jsonl','article_splits.jsonl'):
        if not (directory/f).exists():raise ValueError(f'Incomplete output: {directory/f}')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--goal',type=int,default=100)
    p.add_argument('--min-length',type=int,default=20)
    p.add_argument('--seed',type=int,default=20260915,help='Substitution RNG seed; article split remains fixed')
    p.add_argument('--name',help='New output suffix; default is the goal')
    args=p.parse_args()
    if args.goal<1:p.error('goal must be positive')
    if not 1<=args.min_length<=25:p.error('min-length must be 1..25')
    name=args.name or f'{args.goal}_len{args.min_length}_25'
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):p.error('name must be letters, digits, underscore or hyphen')
    if not (ROOT/'results/character_counts.jsonl').exists():run('analyze_wikipedia.py')
    folder=ROOT/'data/charset_sources';folder.mkdir(parents=True,exist_ok=True)
    for filename,(url,sha) in UNICODE_FILES.items():
        path=folder/filename
        if not path.exists():
            with urlopen(url) as response: data=response.read()
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError(f'Unexpected Unicode data: {url}')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError(f'Unicode checksum mismatch: {path}')
    if not (ROOT/'results/charset/candidates.jsonl').exists():run('classify_characters.py')
    if not (ROOT/'results/charset_refined/targets.jsonl').exists():run('refine_charset.py')
    natural=ROOT/f'data/training_text_direct_{name}';nr=ROOT/f'results/training_text_direct_{name}'
    final=ROOT/f'data/training_text_final_{name}';fr=ROOT/f'results/training_text_final_{name}'
    if not complete(natural,args.goal,args.min_length):run('select_wikipedia_direct.py','--goal',args.goal,'--min-length',args.min_length,'--output',natural,'--report-dir',nr)
    if not (nr/'verification.json').exists():run('verify_training_text.py','--output',natural,'--report-dir',nr)
    if not complete(final,args.goal,args.min_length,args.seed):run('synthesize_shortfalls.py','--base',natural,'--goal',args.goal,'--seed',args.seed,'--output',final,'--report-dir',fr)
    if not (fr/'verification.json').exists():run('verify_training_text.py','--output',final,'--report-dir',fr)
    if not (final/'train.txt').exists():raise ValueError('Verified plaintext output missing')
    summary=json.loads((final/'summary.json').read_text())
    verification=json.loads((fr/'verification.json').read_text())
    (fr/'report.md').write_text(
        f"# 完成した学習テキスト\n\n合計 {summary['samples']:,}件。自然文 {summary['natural_samples']:,}件、置換合成 {summary['synthetic_samples']:,}件。\n\n"
        f"全 {summary['target_entries']:,}字種で目標 {args.goal}件以上。長さ{args.min_length}〜25文字、完全重複なし。\n\n"
        f"置換乱数シード {args.seed}。漢字は漢字、その他は同じUnicode一般カテゴリの文字に置換。\n"
        "元テキストは学習用Wikipedia記事から選択済みの自然文集合からランダムに採る。\n"
        "各合成行にbase_text・base_sample_id・replacementsを記録。元記事のstart/endは置換前の原文位置。\n\n"
        f"全件検証済み。原文照合 {verification['source_substrings_independently_verified']:,}件（合成全件を含む）。\n\n"
        "train.txt: 1行1件の統合データ。train.jsonl: 出典・置換履歴付き。synthetic.jsonl: 合成分のみ。\n",
        encoding='utf-8')
    print(f'Complete: {final / "train.txt"}',flush=True)


if __name__=='__main__':main()
