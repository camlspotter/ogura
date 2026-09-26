"""Inspect quote glyphs with the actual training renderer; no recognition model."""

import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont
from ogura.textrec.text_common import ROOT
from ogura.textrec.training.render import Sample,RenderParams,render_sample,normalized_text,font_characters
from ogura.textrec.training.aliases import CharacterAliases


def main():
    fonts=(Path(__file__).resolve().parents[2] / 'corpus') / 'fonts';output=ROOT/'datasets/font_candidates/quote-review';output.mkdir(parents=True,exist_ok=True)
    jp=['NotoSansCJKjp-Regular.otf','NotoSansCJKjp-Bold.otf','NotoSerifCJKjp-Regular.otf','NotoSerifCJKjp-Bold.otf','NotoSansCJKjp-Light.otf','ZenMaruGothic-Light.ttf','ZenMaruGothic-Regular.ttf','MPLUSRounded1c-Thin.ttf','MPLUSRounded1c-Light.ttf','KosugiMaru-Regular.ttf']
    text='“Ab” ‘Ab’ "Ab" \'Ab\' ＡA ｶﾞガ ～~〜'
    aliases=CharacterAliases.read(ROOT/'config/character_aliases_quotes.json')
    coverage={name:{c:ord(c) in font_characters(str(fonts/name)) for c in '“”‘’"\''} for name in jp+['Tinos-Regular.ttf','Arimo[wght].ttf']}
    report=dict(text=text,expected_label=aliases.normalize(text),coverage=coverage,rows=[])
    for page,western in enumerate([None,'Tinos-Regular.ttf','Arimo[wght].ttf'],1):
        images=[]
        for name in jp:
            for size in (40,28):
                p=RenderParams(str(fonts/name),font_size=size,western_font_path=str(fonts/western) if western else None)
                rendered=normalized_text(text,p)
                assert all(c in rendered for c in '“”‘’')
                im=render_sample(Sample(text,p))
                images.append((f'{name}  {size}px',im))
                report['rows'].append(dict(font=name,western=western,size=size,rendered=rendered,label=aliases.normalize(rendered)))
        width=max(1050,max(im.width for _,im in images)+24)
        sheet=Image.new('RGB',(width,120+len(images)*78),'#eeeeee');d=ImageDraw.Draw(sheet)
        f=ImageFont.truetype(str(fonts/jp[0]),20)
        d.text((12,8),'Japanese only' if western is None else 'Mixed: '+western,font=f,fill='black')
        d.text((12,40),'Source: '+text,font=f,fill='black')
        d.text((12,72),'Label: '+aliases.normalize(text),font=f,fill='black')
        for i,(label,im) in enumerate(images):
            y=120+78*i;d.text((12,y),label,font=f,fill='black');sheet.paste(im,(12,y+26))
        sheet.save(output/f'quotes-{page}.png')
    (output/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(output)
    print(json.dumps(coverage,ensure_ascii=False))


if __name__=='__main__':main()
