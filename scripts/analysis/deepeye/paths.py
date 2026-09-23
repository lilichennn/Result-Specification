"""Shared CLI paths; saved records and metric algorithms live in each tool."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS = ROOT / 'outputs' / 'analysis' / 'deepeye'


def add_analysis_argument(parser):
    parser.add_argument('--analysis-dir', type=Path, default=DEFAULT_ANALYSIS,
                        help='Directory containing GROUP/offline.json and evaluation.sqlite3; '
                             'derived reports are created here (default: outputs/analysis/deepeye).')


def analysis_directory(parser):
    add_analysis_argument(parser)
    return parser.parse_args().analysis_dir.resolve()
