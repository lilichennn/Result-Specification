from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting existing variables."""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_env(Path(__file__).with_name(".env"))

    base_url = os.environ["DASH_BASE_URL"].rstrip("/")
    api_key = os.environ["DASH_API_KEY"]
    model = os.environ["DASH_MODELS"]

    def request_once(request_id: int) -> tuple[int, str]:
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Request {request_id}: report your model name in few words.",
                    }
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
        return request_id, result["choices"][0]["message"]["content"]

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(request_once, request_id) for request_id in range(1, 6)]
        for future in as_completed(futures):
            try:
                request_id, content = future.result()
                print(f"[{request_id}] success: {content}")
            except Exception as exc:
                print(f"[failed] {type(exc).__name__}: {exc}")
    print(f"Total elapsed: {time.monotonic() - started:.2f}s")


if __name__ == "__main__":
    main()
