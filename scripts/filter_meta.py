"""Filter preprocessed metadata using RC, Q/Hint, or their combination."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from result_contract.rc.filter import (
    apply_filter,
    build_filter_messages,
    parse_filter_response,
)
from result_contract.rc import MODEL_ALIASES, call_model
from scripts.generate_rc import _load_instances, _load_metadata


INITIAL_CONCURRENCY = 50
MAX_CONCURRENCY = 2000
CONCURRENCY_GROWTH_MIN = 40
CONCURRENCY_GROWTH_MAX = 60
MAX_ATTEMPTS = 3


class Progress:
    def __init__(self, ids: list[str], results: dict):
        self.ids, self.results = ids, results
        self.prompts = 0
        self.tty = sys.stdout.isatty()
        self.bar = ""
        self.last_update = 0.0
        self.tick(force=True)

    def tick(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last_update >= 5:
            self.last_update = now
            success = sum(
                self.results.get(index, {}).get("status", {}).get("success") is True
                for index in self.ids
            )
            failed = sum(
                self.results.get(index, {}).get("status", {}).get("success") is False
                for index in self.ids
            )
            self.bar = (
                f"instances={len(self.ids)} | prompts={self.prompts} | "
                f"success={success} | fail={failed} | "
                f"empty={len(self.ids) - success - failed}"
            )
            if self.tty:
                print("\r\033[2K" + self.bar, end="", flush=True)
            else:
                print(self.bar, flush=True)

    def message(self, text: str) -> None:
        if self.tty:
            print("\r\033[2K" + text, flush=True)
            print(self.bar, end="", flush=True)
        else:
            print(text, flush=True)

    def log(self, index: str, stage: str, success: bool, reason: str = "") -> None:
        label = "success" if success else "fail"
        if self.tty:
            label = ("\033[32m" if success else "\033[31m") + label + "\033[0m"
        suffix = "｜" + " ".join(reason.splitlines()) if reason else ""
        if stage == 'result':
            self.message(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | "
                f"{index:>5} | filter | {label}{suffix}"
            )

    def close(self) -> None:
        self.tick(force=True)
        if self.tty:
            print()


def save_results(path: Path, results: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def failed_record(exc: Exception) -> dict:
    return {"result": None, "status": {"success": False, "reason": f"{type(exc).__name__}: {exc}"}}


def request_filter(messages: list, metadata: list, llm: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_model(messages, llm=llm)
            result = apply_filter(metadata, parse_filter_response(raw))
            return {"result": result, "status": {"success": True, "reason": ""}}
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt)
    assert last_error is not None
    return failed_record(last_error)


def run(dataset: str, split: str, mode: str | None, llm: str) -> None:
    for value in (dataset, split):
        if not isinstance(value, str) or not value or value in (".", "..") or "/" in value or "\\" in value:
            raise ValueError("dataset and split must be directory-name components")
    if mode not in (None, "rc", "qh"):
        raise ValueError("mode must be rc, qh, or omitted")
    group = f"{dataset}_{split}"
    root = CODE_ROOT / "data" / group
    instances = _load_instances(root / f"{group}.json")
    ids = [str(instance["index"]) for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("Instance indices must remain unique as JSON keys")
    metadata = _load_metadata(root / "meta", {row["db_id"] for row in instances})
    rc_by_index = {}
    if mode != "qh":
        records = json.loads((root / "rc.json").read_text(encoding="utf-8"))
        for record in records:
            index = str(record["index"])
            if index in rc_by_index:
                raise ValueError(f"Duplicate RC index: {index}")
            rc_by_index[index] = record

    output = CODE_ROOT / "outputs" / "schema_filter" / group / (
        f"filtered_meta_{mode}.json" if mode else "filtered_meta.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if not isinstance(results, dict):
        raise ValueError(f"Expected an instance-indexed JSON object: {output}")
    results = {index: results[index] for index in ids if index in results}
    succeeded = lambda index: results.get(index, {}).get("status", {}).get("success") is True
    progress = Progress(ids, results)
    try:
        # Build all prompts before issuing any model requests.
        prompts = {}
        for instance, index in zip(instances, ids):
            try:
                if mode != "qh":
                    record = rc_by_index[index]
                    if record["db_id"] != instance["db_id"]:
                        raise ValueError("RC database does not match instance")
                    if record.get("round3_status") != "succeeded":
                        raise ValueError("Round-3 RC is unavailable")
                    guidance = {"rc": record["rc_round3"]}
                    if mode is None:
                        guidance.update(
                            question=instance["question"], evidence=instance["evidence"]
                        )
                else:
                    guidance = {"question": instance["question"], "evidence": instance["evidence"]}
                messages = build_filter_messages(metadata[instance["db_id"]], **guidance)
                progress.prompts += 1
                progress.log(index, "prompt", True)
                if not succeeded(index):
                    prompts[index] = (messages, metadata[instance["db_id"]])
            except Exception as exc:
                progress.log(index, "prompt", False, str(exc))
                if not succeeded(index):
                    results[index] = failed_record(exc)
                    save_results(output, results)
            progress.tick()

        pending = iter(prompts)
        active = {}
        exhausted = False
        concurrency = INITIAL_CONCURRENCY
        last_growth = time.monotonic()
        last_report = time.monotonic()
        progress.message(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | "
            f"current concurrency={concurrency}"
        )
        with ThreadPoolExecutor(
            max_workers=max(1, min(MAX_CONCURRENCY, len(prompts)))
        ) as executor:
            while active or not exhausted:
                now = time.monotonic()
                growth_steps = int(now - last_growth)
                for _ in range(growth_steps):
                    concurrency = min(
                        MAX_CONCURRENCY,
                        concurrency
                        + random.randint(
                            CONCURRENCY_GROWTH_MIN,
                            CONCURRENCY_GROWTH_MAX,
                        ),
                    )
                if growth_steps:
                    last_growth += growth_steps

                while not exhausted and len(active) < concurrency:
                    index = next(pending, None)
                    if index is None:
                        exhausted = True
                        break
                    future = executor.submit(request_filter, *prompts[index], llm)
                    active[future] = index
                if not active:
                    break

                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    index = active.pop(future)
                    record = future.result()
                    results[index] = record
                    save_results(output, results)
                    success = record["status"]["success"]
                    progress.log(index, "result", success, record["status"]["reason"])
                progress.tick()
                if time.monotonic() - last_report >= 10:
                    last_report = time.monotonic()
                    progress.message(
                        f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | "
                        f"current concurrency={concurrency} | active={len(active)}"
                    )
        save_results(output, results)
    finally:
        progress.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--mode",
        choices=("rc", "qh"),
        help="Omit to filter with question, hint, and RC together",
    )
    parser.add_argument("--llm", required=True, choices=MODEL_ALIASES)
    args = parser.parse_args()
    for value in (args.dataset, args.split):
        if not value or value in (".", "..") or "/" in value or "\\" in value:
            parser.error("dataset and split must be directory-name components")
    run(args.dataset, args.split, args.mode, args.llm)


if __name__ == "__main__":
    main()
