from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

sys.path = [entry for entry in sys.path if entry not in {"", str(SCRIPTS_DIR)}]
sys.path.insert(0, str(PROJECT_ROOT))

loaded_benchmark = sys.modules.get("benchmark")
if loaded_benchmark is not None and str(getattr(loaded_benchmark, "__file__", "")).startswith(str(SCRIPTS_DIR)):
    del sys.modules["benchmark"]
