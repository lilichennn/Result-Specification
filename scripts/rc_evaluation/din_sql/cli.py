"""Single DIN entry point. Preparation/evaluation never call a language model."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dotenv import load_dotenv
from scripts.baseline_adapters.din_sql.inputs import prepare_inputs, load_config, DinSettings, digest
from scripts.baseline_adapters.din_sql.records import DinRecords, hydrate_batch, read_json

CODE_ROOT = Path(__file__).resolve().parents[3]


def prepare(config_path, batch_id, *, code_root=CODE_ROOT, groups=None):
    if not batch_id or Path(batch_id).name != batch_id or batch_id in ('.','..'):
        raise ValueError('batch-id must be one directory name')
    config = load_config(config_path)
    if groups:
        config['groups'] = [g for g in config['groups'] if g['name'] in groups]
        if not config['groups']:
            raise ValueError('No selected groups')
    root = code_root/'baselines_reproduce/din_sql/batches'/batch_id
    if (root/'prepared/import_report.json').exists():
        manifest = read_json(root/'manifest.json')
        if manifest['config'] != config:
            raise ValueError('Existing batch configuration differs')
        return {'batch':str(root),'already_prepared':True,'import':read_json(root/'prepared/import_report.json')}
    prepared = prepare_inputs(config,code_root)
    manifest = {'format':'din-sql-v1','batch_id':batch_id,'config':config,
                'settings':asdict(DinSettings(**config.get('settings',{}))),
                'inputs_fingerprint':digest(prepared.identities),
                'groups':{g['name']:{'ids':[k.question_id for k in prepared.tasks if k.group==g['name']]}
                          for g in config['groups']}}
    with DinRecords(root,manifest) as records:
        report = hydrate_batch(root,prepared,records)
    return {'batch':str(root),'import':report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    pre = sub.add_parser('prepare')
    pre.add_argument('--config',type=Path,default=CODE_ROOT/'config/din_sql/experiment.json')
    pre.add_argument('--batch-id',required=True)
    pre.add_argument('--groups',nargs='+')
    pre.add_argument('--env-file',type=Path,default=CODE_ROOT/'config/.env')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--batch',type=Path,required=True)
    ev.add_argument('--groups',nargs='+',default=['all'])
    ev.add_argument('--env-file',type=Path,default=CODE_ROOT/'config/.env')
    args = parser.parse_args(argv)
    load_dotenv(args.env_file,override=True)
    if args.command=='prepare':
        result = prepare(args.config,args.batch_id,groups=args.groups)
    else:
        from .evaluation import evaluate
        manifest = read_json(args.batch/'manifest.json')
        groups = list(manifest['groups']) if args.groups==['all'] else args.groups
        with DinRecords(args.batch,manifest,read_only=True) as records:
            result = evaluate(args.batch,groups=groups,records=records)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
