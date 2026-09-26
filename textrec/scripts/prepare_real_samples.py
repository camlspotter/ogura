"""Select initial OCR line crops from local PDFs using Poppler (no text re-rendering)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

SOURCES = ('public_document_ministry00045.pdf', 'public_document_ministry02212.pdf',
           'kouhou01552.pdf')
NS = {'h': 'http://www.w3.org/1999/xhtml'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path.home()/'mocrdown/tests/data')
    parser.add_argument('--output', type=Path, default=Path('datasets/real_samples'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for source in SOURCES:
            pdf = args.source_dir/source
            xml = Path(tmp)/'page.html'
            subprocess.run(['pdftotext', '-f', '1', '-l', '1', '-bbox-layout', str(pdf), str(xml)], check=True)
            root = ET.parse(xml).getroot()
            subprocess.run(['pdftoppm', '-f', '1', '-l', '1', '-singlefile', '-r', '300',
                            '-png', str(pdf), str(Path(tmp)/'page')], check=True, capture_output=True)
            page = Image.open(Path(tmp)/'page.png').convert('RGB')
            selected = 0
            for line in root.findall('.//h:line', NS):
                words = line.findall('h:word', NS)
                text = ' '.join(w.text or '' for w in words)
                box = [float(line.attrib[k]) for k in ('xMin','yMin','xMax','yMax')]
                if not (20 <= len(text) <= 85 and box[2]-box[0] > 10*(box[3]-box[1])):
                    continue
                scale = 300/72
                crop_box = (max(0, math.floor((box[0]-2)*scale)), max(0, math.floor((box[1]-1)*scale)),
                            min(page.width, math.ceil((box[2]+2)*scale)), min(page.height, math.ceil((box[3]+1)*scale)))
                name = f'line-{len(rows)+1:02d}.png'
                page.crop(crop_box).save(args.output/name)
                rows.append(dict(image=name, source=source, source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                                 page=1, dpi=300, crop_pixels=crop_box, pdf_text=text,
                                 reference_verified=False))
                selected += 1
                if selected == 4:
                    break
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    sheet = Image.new('RGB', (1500, len(rows)*100+20), 'white')
    draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        im = Image.open(args.output/row['image'])
        im.thumbnail((1380,75))
        sheet.paste(im,(100,i*100+15))
        draw.text((10,i*100+25),row['image'],fill='black')
    sheet.save(args.output/'contact-sheet.png')
    print(f"{len(rows)} lines: {args.output}")
    for r in rows:
        print(r['image'],r['source'],r['pdf_text'])


if __name__ == '__main__':
    main()
