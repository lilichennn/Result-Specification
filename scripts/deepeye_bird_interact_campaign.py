"""Checkout-independent entry point for the rolling DeepEye campaign."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'baselines/DeepEye-SQL'):
    sys.path.insert(0, str(directory))

from scripts.rc_evaluation.deepeye.campaign.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
