"""Independent candidate-SQL evaluation; voting transformations are not gold evaluation."""

import json

from scripts.rc_evaluation.deepeye.comparison import compare_results


def evaluate_candidate(candidate: dict, binding: dict, *, execute, cache) -> dict:
    """Compare original SQL using a caller-owned, short-lived execution cache.

    ``execute(database, sql)`` returns the DAIL execution record. The caller
    bounds SQL duration/resources and the cache lifetime (export uses one
    question at a time). Without a physical database_version, historical vote
    results are never reused. The cache is only valid within this observation.
    Returned evidence is compact; full typed tables stay out of mode exports.
    """
    database = binding['database']
    identity = json.dumps([database, binding.get('database_version')], sort_keys=True)

    def run(sql):
        key = (identity, sql)
        if key not in cache:
            cache[key] = execute(database, sql)
        return cache[key]

    reference = run(binding['reference_sql'])
    sql = candidate.get('candidate_sql')
    vote = candidate.get('vote_execution')
    reuse = (isinstance(vote, dict) and binding.get('database_version') is not None
             and vote.get('database_version') == binding['database_version']
             and vote.get('database') == database and vote.get('sql') == sql)
    predicted = vote if reuse else run(sql)
    def comparison_shape(result):
        return {'result_type': result.get('status'), 'result_cols': result.get('columns'),
                'result_rows': result.get('rows')}
    comparison = compare_results(comparison_shape(predicted), comparison_shape(reference))
    status = ('reference_error' if reference.get('status') != 'success' else
              'prediction_error' if predicted.get('status') != 'success' else
              'comparable' if comparison['comparable'] else 'unknown')
    return {'candidate_id': candidate['candidate_id'], 'candidate_sql': sql,
            'status': status, **comparison, 'vote_reused': reuse,
            'vote_execution_ref': candidate.get('vote_execution_ref') if reuse else None,
            'prediction_execution_id': predicted.get('execution_id'),
            'reference_execution_id': reference.get('execution_id'),
            'prediction_status': predicted.get('status'), 'reference_status': reference.get('status'),
            'prediction_error': predicted.get('error'), 'reference_error': reference.get('error')}
