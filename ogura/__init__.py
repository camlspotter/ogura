"""Detection package and transitional recognition imports from the repository root."""
from pathlib import Path

_recognition = Path(__file__).resolve().parent.parent / 'textrec' / 'ogura'
__path__.extend([str(_recognition), str(_recognition / 'textrec')])
