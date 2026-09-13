"""Explicit offline configure/status and opt-in campaign execution."""
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('configure', 'status', 'run', 'resume', 'pause', '_worker'):
        sub = commands.add_parser(name)
        sub.add_argument('--campaign-dir', type=Path, required=True)
        if name == 'configure':
            sub.add_argument('--env-file', type=Path, required=True)
            sub.add_argument('--workload', type=Path)
            for field in ('precompute-dir', 'few-shot-source', 'rc-lite', 'rc-full'):
                sub.add_argument('--' + field, type=Path)
            sub.add_argument('--tail-fraction', type=float, default=.8)
            sub.add_argument('--poll-seconds', type=float, default=60)
            sub.add_argument('--item', dest='item_keys', action='append')
        elif name == '_worker':
            sub.add_argument('--job-id', required=True)
            sub.add_argument('--token', required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from . import controller
        if args.command == 'configure':
            from .configuration import configure
            result = configure(args)
        elif args.command == '_worker':
            from .supervisor import run_worker
            return run_worker(args.campaign_dir, args.job_id, args.token)
        elif args.command in ('run', 'resume'):
            result = controller.run(args.campaign_dir, resume=args.command == 'resume')
        else:
            result = getattr(controller, args.command)(args.campaign_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        # Exception strings can contain credentials from third-party parsing.
        print(json.dumps({'error_type': type(exc).__name__, 'message': 'Campaign operation failed; inspect campaign status and job logs.'}))
        return 2
