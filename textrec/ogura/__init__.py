"""OCR packages: textrec and textdet.

Legacy ogura.<recognition module> imports and -m commands remain supported.
New code should use ogura.textrec; data/checkpoint locations are unchanged.
"""
from pathlib import Path

__path__.append(str(Path(__file__).parent / 'textrec'))
