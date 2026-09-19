"""Local preparation and bounded diagnostic preflight for DAIL."""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

from scripts.baseline_adapters.dail_sql.config import DailSettings, TaskKey
from scripts.baseline_adapters.dail_sql.records import DailRecords
from scripts.baseline_adapters.dail_sql.transport import GroupRequester, classify_error
from scripts.baseline_adapters.shared.transport import RequestDispatcher, RequestLimits


def preflight(args, environment):
    settings = DailSettings()
    required = ("DASH_API_KEY", "DASH_BASE_URL", "DASH_MODELS")
    if any(not environment.get(name, "").strip() for name in required):
        raise ValueError("Missing required model environment configuration")
    model = environment["DASH_MODELS"].strip()
    if "," in model or model.startswith("["):
        raise ValueError("Preflight requires exactly one configured model")
    messages = [{"role": "user", "content": "Return only SQL for the number one: SELECT 1"}]
    root = Path(args.output).resolve() / "preflight" / args.mode
    manifest = {"batch_id": "preflight", "purpose": "diagnostic_preflight", "mode": args.mode,
                "model": model, "settings": asdict(settings), "messages": messages,
                "groups": {"preflight": {"ids": ["sql-one"]}}}
    # Manifest equality checks model/settings/messages before opening transport.
    with DailRecords(root, manifest) as records:
        key = TaskKey("preflight", "preflight", "sql-one")
        version = args.version_id or records.begin_version(key)
        if records.version_key(version) != key:
            raise ValueError("Preflight version identity mismatch")
        dispatcher = RequestDispatcher(RequestLimits(request_limit=5, http_connections=5,
            request_timeout=settings.request_timeout_seconds),
            fatal_policy=lambda error: classify_error(error)["pause"])
        try:
            client = dispatcher.make_client(api_key=environment["DASH_API_KEY"],
                                            base_url=environment["DASH_BASE_URL"])
            requester = GroupRequester(dispatcher, client, settings, records)
            operation = requester.preflight_sample if args.mode == "single" else requester.generate
            result = asyncio.run(operation(version_id=version, round_execution_id="preflight-" + args.mode,
                                           model=model, messages=messages))
            event = records.append(version, "preflight_result", {"mode": args.mode, "result": result})
            return {"status": result["status"], "mode": args.mode, "model": model,
                    "version_id": version, "result_event_id": event, "records_dir": str(root),
                    "request_attempt_ids": result["request_attempt_ids"],
                    "success_usage": result["success_usage"], "error": result["error"]}
        finally:
            dispatcher.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="Isolated single-sample or five-sample HTTP check")
    check.add_argument("--mode", choices=("single", "group"), default="single")
    check.add_argument("--output", required=True, type=Path)
    check.add_argument("--version-id", help="Resume exactly this diagnostic version")
    check.add_argument("--env-file", type=Path, help="Explicit dotenv source; environment wins")
    prep = commands.add_parser("prepare", help="Incrementally prepare local retrieval assets")
    inputs = prep.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--config", type=Path)
    prep.add_argument("--resources", type=Path, required=True, help="Explicit verified local resource-manifest JSON")
    prep.add_argument("--output", type=Path, required=True, help="Parent directory; result.directory identifies the immutable prepared version")
    prep.add_argument("--encoder-device", choices=("cpu", "mps", "cuda"), default="cpu")
    prep.add_argument("--groups", nargs="+", help="Prepare selected groups in the same canonical cache")
    status = commands.add_parser("status", help="Inspect preparation or compact batch state")
    location = status.add_mutually_exclusive_group(required=True)
    location.add_argument("--preparation", type=Path, help="Exact result.directory returned by prepare")
    location.add_argument("--batch", type=Path)
    for command in ('run', 'smoke'):
        run = commands.add_parser(command)
        run.add_argument('--config', type=Path, required=True)
        run.add_argument('--prepared', type=Path, required=True)
        run.add_argument('--env-file', type=Path, required=True)
        run.add_argument('--groups', nargs='+', default=['all'])
        run.add_argument('--output', type=Path, help='Parent directory for new batch')
        run.add_argument('--request-limit', type=int)
        run.add_argument('--sql-workers', type=int)
        if command == 'run':
            run.add_argument('--batch-id', required=True)
        else:
            run.add_argument('--per-group', type=int, default=2)
            run.add_argument('--targets', type=Path)
    for command in ('resume', 'rerun', 'export'):
        operation = commands.add_parser(command)
        operation.add_argument('--batch', type=Path, required=True)
        if command == 'rerun':
            operation.add_argument('--targets', type=Path, required=True)
        if command == 'export':
            operation.add_argument('--evaluation-profile', choices=('batch', 'deepeye'), default='deepeye',
                                   help='Post-hoc SQL limits/scheduling only; never changes frozen inference settings')
            operation.add_argument('--output', type=Path,
                                   help='Compact export parent; defaults to BATCH/exports/compact')
            operation.add_argument('--seed', type=Path,
                                   help='Optional interrupted uncompressed export whose SQL results are imported once')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare' or args.command == 'status' and args.preparation:
            from scripts.baseline_adapters.dail_sql.preparation import prepare, status_preparation
            if args.command == "prepare":
                resources = json.loads(args.resources.read_text(encoding="utf-8"))
                resources["encoder_device"] = args.encoder_device
                if args.groups and args.groups != ['all']:
                    resources["groups"] = [name for value in args.groups for name in value.split(",")]
                manifest_path = args.manifest
                if args.config is not None:
                    from scripts.baseline_adapters.dail_sql.inputs import build_manifest, write_manifest
                    manifest = build_manifest(Path(__file__).resolve().parents[3],
                                              json.loads(args.config.read_text(encoding="utf-8")))
                    manifest_path = write_manifest(args.output, manifest)
                result = prepare(manifest_path, args.output, resources)
            else:
                result = status_preparation(args.preparation)
            print(json.dumps(result, ensure_ascii=False))
            if args.command == "prepare" and args.groups and args.groups != ['all']:
                return 0 if all(result["groups"][name]["status"] == "ready" for name in resources["groups"]) else 1
            return 0 if result["status"] == "ready" else 1
        if args.command != 'preflight':
            from . import campaign
            if args.command in ('run', 'smoke'):
                config = json.loads(args.config.read_text(encoding='utf-8'))
                batch_id = args.batch_id if args.command == 'run' else 'smoke-' + uuid.uuid4().hex
                if Path(batch_id).name != batch_id or batch_id in ('.', '..'):
                    raise ValueError('batch ID must be one directory name')
                config.update(batch_id=batch_id, env_file=str(args.env_file), selected_groups=args.groups)
                resources = dict(config.get('resources', {}))
                if args.command == 'smoke':
                    config.update(purpose='smoke', smoke_per_group=args.per_group)
                    resources.update(request_limit=20, http_connections=20, sql_workers=20)
                    if args.targets:
                        config['targets'] = campaign.read_targets(args.targets, batch_id)
                if args.request_limit is not None:
                    resources.update(request_limit=args.request_limit, http_connections=args.request_limit)
                if args.sql_workers is not None:
                    resources['sql_workers'] = args.sql_workers
                config['resources'] = resources
                parent = args.output or campaign.ROOT / 'baselines_reproduce/dail_sql' / ('smoke' if args.command == 'smoke' else 'batches')
                result = campaign.run_batch(config, args.prepared, parent / batch_id)
            elif args.command == 'resume':
                result = campaign.resume_batch(args.batch)
            elif args.command == 'rerun':
                manifest = json.loads((args.batch / 'manifest.json').read_text(encoding='utf-8'))
                result = campaign.rerun_questions(args.batch, campaign.read_targets(args.targets, manifest['batch_id']))
            elif args.command == 'status':
                result = campaign.status(args.batch)
            else:
                from .compact_reporting import export_compact_current, compact_current_export
                manifest = json.loads((args.batch / 'manifest.json').read_text(encoding='utf-8'))
                values = campaign.environment(manifest.get('env_file'))
                for name in campaign.PG_FIELDS:
                    if values.get(name) is not None:
                        os.environ[name] = values[name]
                publication = export_compact_current(args.batch, evaluation_profile=args.evaluation_profile,
                                                     seed=args.seed, output_root=args.output)
                result = {'status': 'success', 'directory': str(publication),
                          'current': compact_current_export(args.batch, args.output) == publication}
            print(json.dumps(result, ensure_ascii=False))
            return 0 if args.command == 'status' or result['status'] == 'success' else 1
        environment = dict(os.environ)
        if args.env_file is not None:
            from dotenv import dotenv_values
            if not args.env_file.is_file():
                raise ValueError("Requested environment file is unavailable")
            environment = {**dotenv_values(args.env_file), **environment}
        result = preflight(args, environment)
    except Exception as exc:
        # Exception messages and URLs may contain credentials. Persist/print
        # only stable classifications, never keys or raw environment values.
        if args.command in {"prepare", "status"}:
            result = {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}
        else:
            result = {"status": "failed", "error": classify_error(exc)}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
