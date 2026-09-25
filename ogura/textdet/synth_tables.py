"""Deterministic fictional report tables with independently labelled cell paragraphs."""
import html
import random

# These labels are authored fixture text, not factual statistics or source quotations.
TABLE_TEXT = '表集計結果架空の数値区分項目実績備考件数割合平均第一地区第二地区第三地区第四地区第五地区第六地区第七地区受付相談調査支援報告確認継続対応年度末時点対象外未集計合計参考資料'


def render_table(seed, font_size=None, rows=None):
    rng = random.Random(seed + 7241)
    font_size = font_size or [12,14,16,18][(seed // 4) % 4]
    rows = rows or [6,10,14][(seed // 8) % 3]
    border = ['grid','horizontal','outer','none'][seed % 4]
    merged = (seed // 4) % 2 == 0
    background, foreground = [('#e4edf5','#111'),('#244b70','#fff'),('#eeeeee','#111')][seed % 3]
    counter = 0
    def cell(text, tag='td', **attrs):
        nonlocal counter
        counter += 1
        attr = ''.join(f' {k}="{v}"' for k,v in attrs.items())
        # Explicit newlines become visual lines inside one cell; cell boundaries never merge.
        content = '<br>'.join(html.escape(part) for part in text.split('\n'))
        return f'<{tag}{attr}><p data-id="table-cell-{counter}">{content}</p></{tag}>'
    if merged:
        header = '<tr>'+cell('区分','th',rowspan=2)+cell('項目','th',rowspan=2)+cell('実績','th',colspan=3)+cell('備考','th',rowspan=2)+'</tr>'
        header += '<tr>'+''.join(cell(t,'th') for t in ['件数','割合','平均'])+'</tr>'
    else:
        header = '<tr>'+''.join(cell(t,'th') for t in ['区分','項目','件数','割合','平均','備考'])+'</tr>'
    body = []
    regions = ['第一地区','第二地区','第三地区','第四地区','第五地区','第六地区','第七地区']
    for i in range(rows):
        cells = ''
        if not merged or i % 2 == 0:
            cells += cell(regions[(i//2) % len(regions)], **({'rowspan':min(2,rows-i)} if merged else {}))
        cells += cell(['受付','相談','調査','支援','報告','確認'][i % 6])
        cells += cell(f'{rng.randint(0,99999):,}', **{'class':'number'})
        cells += cell(f'{rng.uniform(0,100):.1f}%', **{'class':'number'})
        cells += cell(f'{rng.uniform(-20,150):.2f}', **{'class':'number'})
        cells += cell(['継続対応\n年度末時点','確認','対象外','未集計',''][i % 5])
        body.append('<tr>'+cells+'</tr>')
    rules = {'grid':'th,td{border:1px solid #666}',
             'horizontal':'th,td{border-bottom:1px solid #777}',
             'outer':'table{border:1.5px solid #555}', 'none':''}
    rule = rules[border].replace('th,td', '.table-block th,.table-block td').replace('table{','.table-block table{')
    css = f'''<style>
.table-block{{flex:none;writing-mode:horizontal-tb;margin:0 0 22px;letter-spacing:0;color:#111}}
.table-block table{{width:100%;table-layout:fixed;border-collapse:collapse;font-size:{font_size}px;line-height:1.35}}
.table-block th,.table-block td{{padding:5px 7px;vertical-align:middle;overflow-wrap:anywhere}}
.table-block th{{background:{background};color:{foreground};font-weight:700;text-align:center}}
.table-block td{{text-align:left}}
.table-block td.number{{text-align:right}}
.table-block p{{margin:0;text-indent:0;font-size:inherit;line-height:inherit}}
.table-block .table-caption{{font-size:{font_size+2}px;margin-bottom:8px;text-align:left}}
{rule}
</style>'''
    markup = ('<section class="table-block"><p class="table-caption" data-id="table-caption">'
              '表　集計結果（架空の数値）</p><table><colgroup>'
              '<col style="width:17%"><col style="width:13%"><col style="width:17%">'
              '<col style="width:14%"><col style="width:14%"><col style="width:25%">'
              '</colgroup><thead>'+header+'</thead><tbody>'+''.join(body)+'</tbody></table></section>')
    return css, markup, dict(border=border, merged=merged, rows=rows, columns=6,
                             font_size=font_size, header_background=background,
                             source='authored labels and seeded fictional numbers')
