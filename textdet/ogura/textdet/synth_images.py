"""Place reviewed local assets using Synth-JDoc's existing image element."""
import random
import re
from jinja2 import Template
from .vendor import synth_jdoc_generator as upstream


def image_plan(names, count, seed, height, position, float_fraction=.5):
    if not names:
        raise ValueError('At least one image asset is required')
    rng = random.Random(seed + 19417)
    selected = []
    while len(selected) < count:
        cycle = list(names)
        rng.shuffle(cycle)
        selected.extend(cycle)
    positions = [position if position != 'both' else ('top' if i % 2 == 0 else 'bottom')
                 for i in range(count)]
    rng.shuffle(positions)
    float_count = round(count * float_fraction)
    layouts = ['float'] * float_count + ['block'] * (count - float_count)
    rng.shuffle(layouts)
    return [dict(file=name, position=pos, height=min(480, max(128, round(height * rng.choice((.16, .22, .28))))),
                 alignment=rng.choice(('left', 'center', 'right')), layout=layout,
                 float_side=rng.choice(('inline-start', 'inline-end')))
            for name, pos, layout in zip(selected[:count], positions, layouts)]


def add_image(markup, style, config):
    image_style = dict(style, is_vertical=False, show_title=False)
    blocks = upstream.preprocess_elements([dict(type='image', src=config['src'], span_all=False)], False)
    fragment = Template(upstream.html_template).render(style=image_style, data={}, processed_blocks=blocks)
    figure = re.search(r'<figure\b.*?</figure>', fragment, re.DOTALL).group()
    figure = figure.replace('class="normal"', 'class="normal asset-figure"').replace('data-id="0"', 'data-id="asset"')
    if config.get('layout', 'block') == 'float':
        # Keep the image inside the first text column, so pagination retains it.
        # Pixel size is resolved against the actual column after fonts load.
        css = ('<style>.asset-figure{'
               f'float:{config["float_side"]};'
               'writing-mode:inherit;box-sizing:border-box;break-inside:avoid;'
               'padding:0!important;margin:10px!important;background:transparent!important;}'
               '.asset-figure .content-image{width:100%!important;height:auto!important;'
               'max-height:none!important;max-width:100%!important;margin:0!important;}</style>')
        markup = markup.replace('<div class="content-body">', '<div class="content-body">' + figure, 1)
        return markup.replace('</head>', css + '</head>')
    margin = {'left': '0 auto 0 0', 'center': '0 auto', 'right': '0 0 0 auto'}[config['alignment']]
    css = ('<style>.asset-figure{flex:none;writing-mode:horizontal-tb!important;'
           'width:100%!important;height:auto!important;padding:8px!important;margin:8px 0!important;'
           'background:transparent!important;}'
           f'.asset-figure .content-image{{height:{config["height"]}px!important;width:auto;'
           f'max-height:none!important;max-width:100%;object-fit:contain;margin:{margin}!important;}}'
           '</style>')
    if style['is_vertical']:
        # In a vertical page the body's flex column runs from right to left.
        # Reserve a side band along that axis instead of consuming its full width.
        align = {'left': 'flex-start', 'center': 'center', 'right': 'flex-end'}[config['alignment']]
        css += ('<style>.asset-figure{'
                f'width:{config["height"] + 16}px!important;height:100%!important;'
                f'display:flex;flex-direction:column;justify-content:{align};'
                'margin:0 8px!important;}'
                '.asset-figure .content-image{width:100%;height:auto!important;max-height:100%!important;'
                'margin:0!important;}</style>')
    if config['position'] == 'top':
        markup = markup.replace('</h1>', '</h1>' + figure, 1)
    else:
        markup = markup.replace('</body>', figure + '</body>', 1)
    return markup.replace('</head>', css + '</head>')


def size_float_image(page, target_size):
    """Limit a float to 42% of its column's inline axis, leaving room for text."""
    return page.evaluate('''target => {
      const content = document.querySelector('.content-body');
      const style = getComputedStyle(content);
      const rect = content.getBoundingClientRect();
      const vertical = style.writingMode.startsWith('vertical');
      const columns = parseInt(style.columnCount) || 1;
      const gap = parseFloat(style.columnGap) || 0;
      const inline = vertical ? rect.height : rect.width;
      const size = Math.min(target, Math.floor(((inline - (columns - 1) * gap) / columns) * .42));
      if (size <= 0) throw new Error('No room for floating image');
      const figure = document.querySelector('.asset-figure');
      figure.style.setProperty('width', size + 'px', 'important');
      figure.style.setProperty('height', size + 'px', 'important');
      return size;
    }''', target_size)
