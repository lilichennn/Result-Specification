"""Command line entry point for frozen gold-SQL schema-linking annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runner import PILOT_SEED, load_settings, prepare_pilot, run_store
from .source import TOTAL_TASKS
from .store import AnnotationStore


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE = ROOT / "outputs" / "schema_linking_gold_annotations" / "pilot.sqlite3"
DEFAULT_ENV = ROOT / "config" / ".env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-pilot", help="Freeze the full store with a deterministic pilot-first plan")
    prepare.add_argument("--source-root", type=Path, required=True, help="Offline analysis directory containing the five group directories")
    prepare.add_argument("--store", type=Path, default=DEFAULT_STORE)
    prepare.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    prepare.add_argument("--size", type=int, default=TOTAL_TASKS)
    prepare.add_argument("--seed", type=int, default=PILOT_SEED)

    run = commands.add_parser("run", help="Run or resume bounded annotation requests")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--store", type=Path, default=DEFAULT_STORE)
    run.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    run.add_argument(
        "--task-limit", "--limit", dest="task_limit", type=int,
        help="Process only the first N pending unique inputs",
    )
    run.add_argument("--batch-limit", type=int, help="Process only the first N pending batches")
    run.add_argument("--phase", choices=("pilot", "rest", "all"), default="pilot",
                     help="Run pilot, rest, or both frozen phases in one paced run")

    for name in ("status", "verify"):
        command = commands.add_parser(name, help=f"{name.title()} the append-only annotation store")
        command.add_argument("--store", type=Path, default=DEFAULT_STORE)

    export = commands.add_parser("export", help="Export accepted labels and reports")
    export.add_argument("--store", type=Path, default=DEFAULT_STORE)
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-pilot":
            settings = load_settings(args.env_file)
            result = prepare_pilot(
                args.source_root,
                args.store,
                settings,
                size=args.size,
                seed=args.seed,
            )
        elif args.command == "run":
            settings = load_settings(args.env_file)
            result = run_store(
                args.store,
                args.source_root,
                settings,
                task_limit=args.task_limit,
                batch_limit=args.batch_limit,
                phase=args.phase,
            )
        elif args.command == "status":
            with AnnotationStore.open(args.store) as store:
                result = {"status": "success", "progress": store.status()}
        elif args.command == "verify":
            with AnnotationStore.open(args.store) as store:
                verification = store.verify()
            result = {
                "status": "success" if verification["ok"] else "failed",
                "ok": verification["ok"],
                "errors": verification["errors"],
                "progress": verification["status"],
            }
        else:
            # Reporting is a later task; keep it out of every other CLI path.
            from .reporting import export_annotations

            result = export_annotations(args.store, args.output, scope=args.scope, source_root=args.source_root)
    except Exception as error:
        # Upstream exceptions and URLs can contain credentials. Never print
        # their messages from this model-facing command.
        result = {"status": "failed", "error": {"type": type(error).__name__}}
    print(json.dumps(_json_value(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "success" else 1


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
