"""Preview the seven selected fonts using the actual training renderer."""
import argparse
import json
from pathlib import Path
import sys

# Permit direct execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ogura.download_training_fonts import font_paths
from ogura.text_common import ROOT
from ogura.training.preview import create_preview
from ogura.training.render import font_characters, Vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir', type=Path, default=ROOT/'corpus/fonts')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/font_candidates/training-preview.png')
    parser.add_argument('--text', default='明日の説明会は3月31日です。申込書を持って受付にお越しください。')
    parser.add_argument('--vocabulary', type=Path, default=ROOT/'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl')
    parser.add_argument('--variants', type=int, default=2)
    args = parser.parse_args()
    fonts = font_paths(args.font_dir, include_rounded=True)
    targets = set(Vocabulary.read(args.vocabulary).characters)
    rows = []
    for path in fonts:
        supported = {chr(cp) for cp in font_characters(str(path))}
        rows.append(dict(font=path.name, total=len(targets), covered=len(targets & supported),
                         missing=len(targets-supported)))
    create_preview(args.text, fonts, args.output, variants=args.variants)
    coverage = args.output.with_suffix('.coverage.json')
    coverage.write_text(json.dumps(rows, indent=2)+'\n')
    print(args.output)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
