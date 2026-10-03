"""Make ``tests.fakes`` importable and the package resolvable from a checkout."""

import sys
from pathlib import Path

# Allow `python -m pytest` from the orchard_eval directory without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
