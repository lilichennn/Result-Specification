"""Compatibility entry for BIRD-Interact arguments; implementation is shared."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepeye_run import *  # Existing public imports continue to work.
from scripts.deepeye_run import _build_parser, _validate_run_args, _execute_pipeline


if __name__ == '__main__':
    raise SystemExit(main())
