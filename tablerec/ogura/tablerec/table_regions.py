"""Deterministic text and whitespace masks, excluding structural boundaries."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def regions(sample, font_dir=None):
    recipe = sample['recipe']
    text = Image.new('L', sample['image'].size)
    draw = ImageDraw.Draw(text)
    if recipe.get('background_context') != 'blank':
        path = Path(font_dir)/recipe['font']['name'] if font_dir else Path(recipe['font']['path'])
        for cell in recipe['cells']:
            font = ImageFont.truetype(str(path), cell['font_size'])
            for run in cell['text_runs']:
                x,y = run['xy']
                a,b,c,d = font.getbbox(run['text'],anchor='lt')
                draw.rectangle((x+a-1,y+b-1,x+c+1,y+d+1),fill=255)
    truth = np.stack([np.array(sample[k]) for k in ('horizontal','vertical')])/255
    border = Image.fromarray((truth.max(0)>0).astype('uint8')*255).filter(ImageFilter.MaxFilter(9))
    safe = np.array(border)==0
    text_region = (np.array(text)>0)&safe
    # White background only; avoid treating symbols or shaded cell fills as blank.
    white = (np.array(sample['image']).min(2)>=250)&safe&(np.array(text)==0)
    masks={'text':text_region,'white':white}
    marked=np.zeros_like(safe)
    for role in ('underline','shallow_diagonal'):
        scale=recipe['scale']
        layer=Image.new('L',(sample['image'].width*scale,sample['image'].height*scale))
        brush=ImageDraw.Draw(layer)
        if recipe.get('background_context')!='blank':
            for cell in recipe['cells']:
                for mark in cell.get('marks',[]):
                    if mark.get('role')==role:
                        for segment in mark['segments']:
                            brush.line(tuple(v*scale for v in segment),fill=255,
                                       width=max(1,round(mark['width']*scale)))
        layer=layer.resize(sample['image'].size,Image.Resampling.BOX).filter(ImageFilter.MaxFilter(3))
        masks[role]=(np.array(layer)>0)&safe
        marked |= masks[role]
    masks['white'] &= ~marked
    return truth,masks
