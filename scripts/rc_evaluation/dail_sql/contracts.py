"""Fail-closed selection of the actual, successful third RC round."""

import copy
import json


RC_FIELDS = ("population", "row_grain", "column_role", "derivation",
             "filter_policy", "meta_review")


def select_rc3(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("RC record must be an object")
    if record.get("round3_status") != "succeeded" or record.get("round3_error") not in (None, ""):
        raise ValueError("successful Round3 required; Round2 fallback is forbidden")
    content = record.get("rc_round3")
    if not isinstance(content, dict) or any(not isinstance(content.get(key), str) or not content[key]
                                            for key in RC_FIELDS):
        raise ValueError("Round3 content is missing, duplicate, or incomplete")
    if record.get("index") is None or not record.get("db_id") or not record.get("question"):
        raise ValueError("RC3 source identity is incomplete")
    return {"question_id": str(record["index"]), "original_id": record["index"],
            "database_id": record["db_id"], "question": record["question"],
            "evidence": record.get("evidence", ""), "rc_round3": dict(content)}


def inject_rc3(messages: list[dict], rc3: dict, definition: str) -> list[dict]:
    """Prepend one removable block to the final user message, keeping SELECT tail.

    rc3 is the six-field content returned by select_rc3, not an RC history record.
    The fixed definition is loaded once by the production runner's caller/factory.
    """
    if not isinstance(rc3, dict) or any(not isinstance(rc3.get(key), str) or not rc3[key]
                                        for key in RC_FIELDS):
        raise ValueError("Complete successful RC3 content is required")
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("A fixed RC field definition is required")
    result = copy.deepcopy(messages)
    position = next((i for i in range(len(result) - 1, -1, -1) if result[i].get("role") == "user"), None)
    if position is None or not isinstance(result[position].get("content"), str):
        raise ValueError("RC injection requires a text user message")
    # JSON escapes preserve literal field values without allowing a value to
    # close the fixed block used for exact native-prompt recovery.
    content = json.dumps({key: rc3[key] for key in RC_FIELDS}, ensure_ascii=False, indent=2).replace("<", "\\u003c")
    block = ("<result_contract>\n" + definition.strip() + "\n\n<rc_round3>\n" + content
             + "\n</rc_round3>\n\nWhen generating the SQL query, consult this RC.\n</result_contract>\n\n")
    result[position]["content"] = block + result[position]["content"]
    return result
