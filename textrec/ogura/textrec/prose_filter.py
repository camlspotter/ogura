"""Conservative Japanese-prose heuristic; not a semantic language classifier."""
import re

KANA_RUN=re.compile(r'[ぁ-ゖァ-ヺㇰ-ㇿー゚゙]{8,}')
PROSE=re.compile(r'である|であった|です|ます|でした|ました|する|した|され|して|という|として|によ[るりって]|から|まで|など|[一-鿿ァ-ヺ][のにはをがでとへも][一-鿿ぁ-ゖァ-ヺ]')
HIRAGANA=re.compile(r'[ぁ-ゖ]')
TABLE_TITLE=re.compile(r'文字コード表|文字一覧|字形一覧|Unicode一覧|ユニコード一覧')


def is_prose(text,title=''):
    if TABLE_TITLE.search(title):return False
    if re.search(r'(?:[^\W\d_]\s+){5}[^\W\d_]',text):return False
    for match in KANA_RUN.finditer(text):
        run=match.group()
        small=sum(c in 'ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ' for c in run)
        if small>=4 or (len(run)>=10 and len(set(run))/len(run)>=0.9):return False
    # Require Japanese function-word evidence within the excerpt itself, rather
    # than treating an explanatory sentence elsewhere on the page as sufficient.
    return len(HIRAGANA.findall(text))>=3 and PROSE.search(text) is not None
