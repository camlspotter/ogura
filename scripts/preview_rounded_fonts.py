"""Compare candidate rounded fonts at fixed sizes with the training renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont
from ogura.text_common import ROOT
from ogura.training.render import RenderParams, Sample, render_sample, font_characters, Vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', default='健康診査の受付は3月31日まで。母子健康手帳をお持ちください。')
    parser.add_argument('--output', type=Path, default=ROOT/'datasets/font_candidates/rounded-comparison.png')
    args = parser.parse_args()
    directory = ROOT/'corpus/fonts'
    paths = [directory/'ZenMaruGothic-Light.ttf', directory/'ZenMaruGothic-Regular.ttf',
             directory/'MPLUSRounded1c-Thin.ttf',
             directory/'MPLUSRounded1c-Light.ttf',
             directory/'KosugiMaru-Regular.ttf']
    targets = set(Vocabulary.read(ROOT/'datasets/final_50_len20_25_hiragana_mix5/targets.jsonl').source_characters)
    rows, report = [], []
    for path in paths:
        supported = {chr(cp) for cp in font_characters(str(path))}
        missing = set(args.text) - supported
        if missing: raise ValueError(f'Sample contains unsupported characters in {path.name}: {missing}')
        report.append(dict(font=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           covered=len(targets & supported), total=len(targets)))
        for size in (40, 28):
            image = render_sample(Sample(args.text, RenderParams(str(path),font_size=size)))
            rows.append((f'{path.stem}   {size}px / height 48px',image))
    label = ImageFont.truetype(str(directory/'NotoSansCJKjp-Regular.otf'),18)
    sheet = Image.new('RGB',(max(im.width for _,im in rows)+24,len(rows)*84+12),'#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for i,(title,im) in enumerate(rows):
        y=6+i*84
        draw.text((12,y),title,font=label,fill='black')
        sheet.paste(im,(12,y+26))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps(dict(text=args.text,fonts=report),ensure_ascii=False,indent=2)+'\n')
    print(args.output)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':main()
