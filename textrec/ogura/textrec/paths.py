"""Recognition assets live locally; raw corpus and fonts are shared with detection."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = PROJECT_ROOT.parent / 'corpus'
