"""Read-only diagnostic of non-RC Selection input differences; no pipeline edits."""
from collections import Counter
import json

from .analyze import HERE, checked, read_db, restore_jsonable, sha, write_new
from .paths import analysis_directory
import argparse


def audit(group, saved):
    data = json.loads((HERE / group / 'offline.json').read_bytes())
    selected = [r for r in saved['items'] if r['active'] and r['same_list_order']]
    wanted = {r[s + '_attempt_id'] for r in selected for s in ('native', 'rc')}
    inputs = {}
    checked_events = 0
    for meta in data['manifests']:
        if meta['condition'] == 'rc' and meta['stage'] != 'sql_selection':
            continue
        with read_db(meta['run_dir'] + '/run.sqlite3') as db:
            manifest = db.execute('select payload_json,payload_checksum from manifest').fetchone()
            checked(manifest[0], manifest[1])
            assert manifest[1] == meta['manifest_sha256']
            for e in db.execute("""select attempt_id,payload_json,payload_checksum from events
                    where kind='component_start' and
                    json_extract(payload_json,'$.component')='selection.pairwise_comparison'"""):
                if e['attempt_id'] not in wanted:
                    continue
                value = restore_jsonable(checked(e['payload_json'], e['payload_checksum']))
                assert e['attempt_id'] not in inputs
                inputs[e['attempt_id']] = value['inputs']
                checked_events += 1
    assert set(inputs) == wanted
    rows = []
    for row in selected:
        a, b = (inputs[row[s + '_attempt_id']] for s in ('native', 'rc'))
        assert a.keys() == b.keys()
        changed = sorted(k for k in a if a[k] != b[k])
        schema_a, schema_b = a['database_schema_profile'], b['database_schema_profile']
        rows.append({'item_key': row['item_key'], 'changed_component_fields': changed,
                     'same_schema_line_multiset': Counter(schema_a.splitlines()) == Counter(schema_b.splitlines()),
                     'native_schema_sha256': sha(schema_a), 'rc_schema_sha256': sha(schema_b),
                     'same_non_rc_request': row['same_non_rc_request'],
                     'same_choice_weights': row['same_choice_weights']})
    return {'n': len(rows), 'source_component_events_checked': checked_events,
            'changed_field_sets': dict(Counter('|'.join(r['changed_component_fields']) or 'none' for r in rows)),
            'same_schema_line_multiset': sum(r['same_schema_line_multiset'] for r in rows),
            'identical_component_but_request_changed': sum(not r['changed_component_fields'] and not r['same_non_rc_request'] for r in rows),
            'items': rows}


if __name__ == '__main__':
    HERE = analysis_directory(argparse.ArgumentParser(description=__doc__))
    source = (HERE / 'selection_metrics.json').read_bytes()
    result = {'source_metrics_sha256': sha(source), 'groups': {}}
    for group, saved in json.loads(source)['groups'].items():
        result['groups'][group] = audit(group, saved)
        print(group, {k: v for k, v in result['groups'][group].items() if k != 'items'}, flush=True)
    destination = HERE / 'selection_prompt_audit.json'
    write_new(destination, result)
    assert json.loads(destination.read_text()) == result
