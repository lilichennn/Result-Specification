"""Observed model usage derived from immutable RunStore trace events."""

from __future__ import annotations

from typing import Any


_API_KINDS = {"api_request", "api_response", "api_error"}
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def sampling_completeness(events):
    """Describe sampling completeness, not native stage completion.

    Absence of group events preserves historical stores' original semantics;
    it does not claim that old calls had the new effective sampling metric.
    """
    groups, sample_results = {}, set()
    for event in events:
        kind, payload = event['kind'], event['payload']
        if kind == 'sample_result':
            _claim_sample_result(event, sample_results)
        if kind not in ('sampling_group_start', 'sampling_group_result'):
            continue
        group_id = payload.get('group_id')
        if not isinstance(group_id, str) or not group_id:
            raise ValueError('sampling group has no identity')
        group_id = (event.get('attempt_id'), group_id)
        if kind == 'sampling_group_start':
            if group_id in groups:
                raise ValueError('duplicate sampling group start')
            target = payload.get('target_n')
            if isinstance(target, bool) or not isinstance(target, int) or target < 1:
                raise ValueError('invalid sampling target')
            groups[group_id] = {'target': target, 'complete': False, 'terminal': False}
        else:
            group = groups.get(group_id)
            if group is None or group['terminal']:
                raise ValueError('sampling group has no unique start')
            group['terminal'] = True
            group['complete'] = (payload.get('complete') is True
                and payload.get('target_n') == group['target']
                and payload.get('success_count') == group['target'])
    failed = sum(not group['complete'] for group in groups.values())
    return {'groups': len(groups), 'incomplete_groups': failed, 'complete': failed == 0}


def _claim_sample_result(event, seen):
    payload = event['payload']
    key = (event.get('attempt_id'), payload['group_id'], payload['sample_index'])
    if key in seen:
        raise ValueError('duplicate sample result within stage attempt')
    seen.add(key)


def _effective_sampling(events):
    results, seen = {}, set()
    for event in events:
        if event['kind'] != 'sample_result':
            continue
        _claim_sample_result(event, seen)
        payload = event['payload']
        identity = (payload['group_id'], payload['sample_index'])
        if identity in results:
            prior = results[identity]
            if (payload.get('restored_from_event') is None or not payload['succeeded']
                    or any(payload.get(key) != prior.get(key) for key in ('result', 'usage', 'response_id'))):
                # Failed samples can be retried using their remaining original
                # allowance, but successful results must only be restored.
                if prior['succeeded'] or event.get('attempt_id') == prior.get('_stage_attempt'):
                    raise ValueError('duplicate sample result')
            if prior['succeeded']:
                continue
        results[identity] = {**payload, '_stage_attempt': event.get('attempt_id')}
    retained = [value for value in results.values() if value['succeeded']]
    known = {field: 0 for field in _TOKEN_FIELDS}
    missing, missing_reasoning, reasoning = 0, 0, 0
    for sample in retained:
        usage = sample.get('usage') or {}
        complete = True
        for field in _TOKEN_FIELDS:
            value = usage.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                known[field] += value
            else:
                complete = False
        missing += not complete
        value = usage.get('reasoning_tokens')
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            reasoning += value
        else:
            missing_reasoning += 1
    return {'known_tokens': known, 'retained_samples': len(retained),
        'samples_missing_usage': missing, 'usage_complete': missing == 0,
        'known_reasoning_tokens': reasoning,
        'reasoning_tokens': None if missing_reasoning else reasoning,
        'samples_missing_reasoning': missing_reasoning,
        'semantics': 'finally_retained_successful_samples_only_v1'}


def _call_id(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"{event.get('kind')} event payload must be a dictionary")
    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"{event.get('kind')} event has no valid call_id")
    return call_id


def observed_usage(store: Any) -> dict[str, Any]:
    """Aggregate provider-reported tokens across every recorded attempt.

    This is an observation ledger, not a billing statement. API failures,
    in-flight calls, and successful responses without complete usage metadata
    make ``usage_complete`` false while retaining every token count that the
    provider did report.
    """

    calls: dict[str, dict[str, Any]] = {}
    reported = {field: 0 for field in _TOKEN_FIELDS}
    requests = 0
    responses = 0
    errors = 0
    responses_missing_usage = 0
    errors_missing_usage = 0
    reported_reasoning = 0
    missing_reasoning = 0

    for event in store.iter_events(kinds=_API_KINDS):
        kind = event["kind"]
        call_id = _call_id(event)
        if kind == "api_request":
            if call_id in calls:
                raise ValueError(f"duplicate API call id: {call_id}")
            calls[call_id] = {
                "attempt_id": event["attempt_id"],
                "terminal": None,
            }
            requests += 1
            continue

        call = calls.get(call_id)
        if call is None:
            raise ValueError(f"{kind} event has no preceding request: {call_id}")
        if call["attempt_id"] != event["attempt_id"]:
            raise ValueError(f"API call crosses attempt boundaries: {call_id}")
        if call["terminal"] is not None:
            raise ValueError(f"API call has duplicate terminal events: {call_id}")
        call["terminal"] = kind

        if kind == "api_error":
            errors += 1
        else:
            responses += 1
        payload = event["payload"]
        response = payload.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        details = usage.get('completion_tokens_details') if isinstance(usage, dict) else None
        reasoning = (details.get('reasoning_tokens') if isinstance(details, dict)
                     else usage.get('reasoning_tokens') if isinstance(usage, dict) else None)
        if isinstance(reasoning, int) and not isinstance(reasoning, bool) and reasoning >= 0:
            reported_reasoning += reasoning
        else:
            missing_reasoning += 1
        complete_usage = isinstance(usage, dict)
        if isinstance(usage, dict):
            for field in _TOKEN_FIELDS:
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    reported[field] += value
                else:
                    complete_usage = False
        if not complete_usage:
            if kind == 'api_error':
                errors_missing_usage += 1
            else:
                responses_missing_usage += 1

    unanswered_requests = sum(call["terminal"] is None for call in calls.values())
    result = {
        "reported_tokens": reported,
        "requests": requests,
        "responses": responses,
        "errors": errors,
        "unanswered_requests": unanswered_requests,
        "responses_missing_usage": responses_missing_usage,
        "usage_complete": (
            errors == 0
            and unanswered_requests == 0
            and responses_missing_usage == 0
        ),
        "semantics": "reported_tokens_only_not_provider_bill",
    }
    sampling_events = list(store.iter_events(kinds={'sampling_group_start', 'sampling_group_result', 'sample_result'}))
    if sampling_events:
        result['sampling'] = sampling_completeness(sampling_events)
        result['effective_sampling'] = _effective_sampling(sampling_events)
        result['unknown_usage_attempts'] = responses_missing_usage + errors_missing_usage + unanswered_requests
        result['known_reported_reasoning_tokens'] = reported_reasoning
        result['reasoning_usage_unknown_attempts'] = missing_reasoning + unanswered_requests
        result['reported_reasoning_tokens'] = None if missing_reasoning + unanswered_requests else reported_reasoning
        result['usage_complete'] = result['unknown_usage_attempts'] == 0
    return result
