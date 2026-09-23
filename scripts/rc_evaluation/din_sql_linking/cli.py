"""CLI for the focused DIN Round-3 RS schema-filtering and Linking campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.baseline_adapters.din_sql.inputs import TaskKey
from .campaign import prepare_batch, run_batch, status, verify_batch


CODE_ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-batch", type=Path, required=True)
    prepare.add_argument("--batch-id", required=True)
    prepare.add_argument("--groups", nargs="+")
    for command in ("run", "resume", "rerun", "status", "verify"):
        child = commands.add_parser(command)
        child.add_argument("--batch", type=Path, required=True)
        if command in ("run", "resume", "rerun"):
            child.add_argument("--env-file", type=Path, default=CODE_ROOT / "config/.env")
        if command == "resume":
            child.add_argument("--all-pending", action="store_true")
        if command == "rerun":
            child.add_argument("--group", required=True)
            child.add_argument("--ids", nargs="+", required=True)
    handoff = commands.add_parser("export-handoff", help="Export the completed unified DIN filter handoff")
    handoff.add_argument("--source-handoff", type=Path, required=True)
    handoff.add_argument("--batch", type=Path, required=True, help="Completed DIN Linking extension batch")
    handoff.add_argument("--annotations", type=Path,
                         default=CODE_ROOT / "data/reference/schema_linking_annotations.jsonl")
    handoff.add_argument("--output", type=Path, required=True)
    handoff.add_argument("--guide", type=Path)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_batch(
            args.source_batch,
            args.batch_id,
            code_root=CODE_ROOT,
            groups=args.groups,
        )
    elif args.command == "status":
        result = status(args.batch)
    elif args.command == "verify":
        result = verify_batch(args.batch)
    elif args.command == "export-handoff":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        from .reporting import export_unified_handoff
        result = {"output": str(export_unified_handoff(args.source_handoff, args.batch,
                        args.annotations, args.output, guide=args.guide))}
    else:
        targets = (
            [TaskKey(args.group, question_id) for question_id in args.ids]
            if args.command == "rerun" else None
        )
        result = run_batch(
            args.batch,
            args.env_file,
            operation=args.command,
            targets=targets,
            all_pending=getattr(args, "all_pending", False),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
