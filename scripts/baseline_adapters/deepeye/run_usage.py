"""Observed model usage derived from immutable RunStore trace events."""

from __future__ import annotations

from typing import Any


_API_KINDS = {"api_request", "api_response", "api_error"}
_TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


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
            continue

        responses += 1
        payload = event["payload"]
        response = payload.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        complete_usage = isinstance(usage, dict)
        if isinstance(usage, dict):
            for field in _TOKEN_FIELDS:
                value = usage.get(field)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    reported[field] += value
                else:
                    complete_usage = False
        if not complete_usage:
            responses_missing_usage += 1

    unanswered_requests = sum(call["terminal"] is None for call in calls.values())
    return {
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
