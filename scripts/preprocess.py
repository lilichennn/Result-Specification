from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from result_contract.data_preprocess import preprocess_bird, preprocess_spider


DEFAULT_DATASET_ROOTS = {
    "bird": Path("BIRD"),
    "spider": Path("Spider"),
}
SUPPORTED_SPLITS = {
    "bird": {"dev"},
    "spider": {"dev", "test"},
    "birdinteract":{},
    "spider2":{}
}


def preprocess_dataset(
    dataset: str,
    split: str,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Preprocess one supported dataset split into its scripts workspace."""
    dataset = dataset.lower()
    split = split.lower()
    _validate_dataset_split(dataset, split)

    resolved_dataset_root = Path(
        dataset_root if dataset_root is not None else DEFAULT_DATASET_ROOTS[dataset]
    )
    resolved_output_dir = Path(
        output_dir
        if output_dir is not None
        else SCRIPT_DIR / f"{dataset}_{split}" / "preprocessed_data"
    )

    if dataset == "bird":
        summary = preprocess_bird(
            bird_root=resolved_dataset_root,
            split=split,
            output_dir=resolved_output_dir,
        )
    else:
        summary = preprocess_spider(
            spider_root=resolved_dataset_root,
            split=split,
            output_dir=resolved_output_dir,
        )

    return {"dataset": dataset, **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a benchmark split for RC generation."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(sorted(SUPPORTED_SPLITS)),
        help="Dataset to preprocess. choose from [bird, spider, birdinteract, spider2]",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("dev", "test", ""),
        help="Dataset split to preprocess.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset root; defaults to BIRD1.0 or Spider1.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the output directory; defaults to "
            "code/scripts/<dataset>_<split>/preprocessed_data."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preprocess_dataset(
        dataset=args.dataset,
        split=args.split,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _validate_dataset_split(dataset: str, split: str) -> None:
    if dataset not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}; expected one of "
            f"{sorted(SUPPORTED_SPLITS)}"
        )
    if split not in SUPPORTED_SPLITS[dataset]:
        raise ValueError(
            f"Unsupported split {split!r} for {dataset!r}; expected one of "
            f"{sorted(SUPPORTED_SPLITS[dataset])}"
        )


if __name__ == "__main__":
    main()
