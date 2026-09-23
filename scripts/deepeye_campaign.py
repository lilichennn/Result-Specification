"""Shared entry point for native DeepEye and stage-specific RC campaigns."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'baselines/DeepEye-SQL'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.rc_evaluation.deepeye.campaign.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
