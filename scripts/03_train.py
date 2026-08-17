"""Phase 3: Train all models (v5-baseline).

Run: python scripts/03_train.py

No Optuna, no ensembles, no Weibull AFT. < 15 minutes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.training.train_all import train

if __name__ == "__main__":
    train()
