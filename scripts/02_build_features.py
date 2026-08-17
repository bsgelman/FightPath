"""Phase 2: Build features from ledger.

Run: python scripts/02_build_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import subprocess

from ufc.io import paths, parquet
from ufc.features.assemble import assemble


def _gitsha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.root(), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")


def main():
    print("=== Phase 2: Feature Engineering ===")
    ledger = parquet.read(paths.processed("ledger"))
    print(f"Loaded ledger: {len(ledger)} rows")
    assemble(ledger, gitsha=_gitsha())


if __name__ == "__main__":
    main()
