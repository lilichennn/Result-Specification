"""Filter preprocessed metadata using RC, Q/Hint, or their combination."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from result_contract.rc.filter import (
    apply_filter,
    build_filter_messages,
    parse_filter_response,
)
from scripts.generate_rc import _load_instances, _load_metadata


class Progress:
    def __init__(self, total: int, results: int):
        self.total, self.results = total, results
        self.prompts = 0
        self.tty = sys.stdout.isatty()
        self.bar = ""
        self.last_update = 0.0
        self.tick(force=True)

    def tick(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last_update >= 10:
            self.last_update = now
            self.bar = (
                f"共 {self.total} instance，已完成 {self.prompts} prompt，"
                f"{self.results} result（含已有成功结果）"
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
        label = "success" if success else "failed"
        if self.tty:
            label = ("\033[32m" if success else "\033[31m") + label + "\033[0m"
        suffix = "｜" + " ".join(reason.splitlines()) if reason else ""
        if stage == 'result':
            self.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜{index}｜【{stage}】｜{label}{suffix}")

    def close(self) -> None:
        self.tick(force=True)
        if self.tty:
            print()


class Concurrency:
    def __init__(self):
        self.limit = 10
        self.successes = self.failures = 0

    def observe(self, success: bool) -> None:
        if success:
            self.failures = 0
            self.successes += 1
            if self.successes == 5:
                self.limit = min(100, self.limit + 5)
                self.successes = 0
        else:
            self.successes = 0
            self.failures += 1
            if self.failures == 3:
                self.limit = max(5, self.limit - 5)
                self.failures = 0


def save_results(path: Path, results: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def failed_record(exc: Exception) -> dict:
    return {"result": None, "status": {"success": False, "reason": f"{type(exc).__name__}: {exc}"}}


def model_config(alias: str) -> dict:
    config = {}
    path = SCRIPT_DIR.parent / "config" / ".env"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip().strip("\"'")
    prefix = config.get(alias)
    if not prefix:
        raise ValueError(f"Missing model alias in .env: {alias}")
    for key in (prefix, prefix + "_API_KEY", prefix + "_BASE_URL"):
        if not config.get(key):
            raise ValueError(f"Missing model configuration key: {key}")
    return {
        "model": config[prefix],
        "api_key": config[prefix + "_API_KEY"],
        "url": config[prefix + "_BASE_URL"].rstrip("/") + "/chat/completions",
    }


def call_model(messages: list, config: dict) -> str:
    request = Request(
        config["url"],
        data=json.dumps({"model": config["model"], "messages": messages, "temperature": 0},
                        ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + config["api_key"], "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body["choices"][0]["message"]["content"]


def request_filter(messages: list, metadata: list, config: dict) -> dict:
    try:
        raw = call_model(messages, config)
        result = apply_filter(metadata, parse_filter_response(raw))
        return {"result": result, "status": {"success": True, "reason": ""}}
    except Exception as exc:
        return failed_record(exc)


def run(dataset: str, split: str, mode: str | None, llm: str) -> None:
    config = model_config(llm)
    root = SCRIPT_DIR / f"{dataset}_{split}"
    preprocessed = root / "preprocessed_data"
    instances = _load_instances(preprocessed / f"{dataset}_{split}.json")
    ids = [str(instance["index"]) for instance in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("Instance indices must remain unique as JSON keys")
    metadata = _load_metadata(preprocessed / "meta", {row["db_id"] for row in instances})
    rc_by_index = {}
    if mode != "qh":
        records = json.loads((root / "rc.json").read_text(encoding="utf-8"))
        for record in records:
            index = str(record["index"])
            if index in rc_by_index:
                raise ValueError(f"Duplicate RC index: {index}")
            rc_by_index[index] = record

    output = root / (f"filtered_meta_{mode}.json" if mode else "filtered_meta.json")
    results = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    if not isinstance(results, dict):
        raise ValueError(f"Expected an instance-indexed JSON object: {output}")
    results = {index: results[index] for index in ids if index in results}
    succeeded = lambda index: results.get(index, {}).get("status", {}).get("success") is True
    progress = Progress(len(ids), sum(succeeded(index) for index in ids))
    try:
        # Build all prompts before issuing any model requests.
        prompts = {}
        for instance, index in zip(instances, ids):
            try:
                if mode != "qh":
                    record = rc_by_index[index]
                    if record["db_id"] != instance["db_id"]:
                        raise ValueError("RC database does not match instance")
                    if record.get("round2_status") != "succeeded":
                        raise ValueError("Round-2 RC is unavailable")
                    guidance = {"rc": record["rc_round2"]}
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
        completed = queue.Queue()
        active = {}
        exhausted = False
        concurrency = Concurrency()
        last_report = time.monotonic()
        progress.message(f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发：10")
        with ThreadPoolExecutor(max_workers=100) as executor:
            while active or not exhausted:
                while not exhausted and len(active) < concurrency.limit:
                    index = next(pending, None)
                    if index is None:
                        exhausted = True
                        break
                    future = executor.submit(request_filter, *prompts[index], config)
                    active[future] = index
                    future.add_done_callback(completed.put)
                if not active:
                    break
                try:
                    future = completed.get(timeout=1)
                except queue.Empty:
                    pass
                else:
                    index = active.pop(future)
                    record = future.result()
                    results[index] = record
                    save_results(output, results)
                    success = record["status"]["success"]
                    concurrency.observe(success)
                    progress.results += int(success)
                    progress.log(index, "result", success, record["status"]["reason"])
                progress.tick()
                if time.monotonic() - last_report >= 30:
                    last_report = time.monotonic()
                    progress.message(
                        f"{datetime.now():%Y-%m-%d %H:%M:%S}｜当前并发：{concurrency.limit}，运行中：{len(active)}"
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
    parser.add_argument("--llm", required=True, choices=("qwen38", "kimik3", "gpt56", "opus48"))
    args = parser.parse_args()
    for value in (args.dataset, args.split):
        if not value or value in (".", "..") or "/" in value or "\\" in value:
            parser.error("dataset and split must be directory-name components")
    run(args.dataset, args.split, args.mode, args.llm)


if __name__ == "__main__":
    main()
