#!/usr/bin/env python3
"""Train a SpectralWaste model. Same flags as ``python -m sw.train``.

Adds ``src/`` to the import path so this works from a fresh clone, before
``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sw.train import main

if __name__ == "__main__":
    main()
