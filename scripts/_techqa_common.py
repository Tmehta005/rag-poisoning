"""
Shared boilerplate for the TechQA wrapper scripts in this directory.

Each ``run_techqa_*.py`` and ``stage_techqa_ibm.py`` etc. needs the same two
setup steps before its real work begins:

  1. Put the repo root on ``sys.path`` so ``from src.<...> import ...`` works
     when the script is invoked as ``python scripts/run_techqa_*.py``.
  2. Best-effort load ``.env`` so ``OPENAI_API_KEY`` etc. are available.

Centralising it removes ~10 lines of boilerplate per wrapper and keeps the
setup consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path


def setup() -> Path:
    """
    Put the repo root on ``sys.path`` and load ``.env`` if present.
    Returns the repo root path.
    """
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv()
    except ImportError:
        pass
    return repo
