"""Characterize frozen pre-C1 code, never the evolving production runtime."""
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace, ModuleType

SOURCE_REVISION = 'be573b1f2ce07d2fd91cd25d2c37da5917c2f56b'


def _frozen_module(name, path):
    root = Path(__file__).resolve().parents[2]
    try:
        source = subprocess.run(['git', 'show', f'{SOURCE_REVISION}:{path}'],
            cwd=root, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Historical characterization source unavailable: {SOURCE_REVISION}:{path}') from error
    module = ModuleType(f'_deepeye_historical_{name}')
    module.__file__ = f'{SOURCE_REVISION}:{path}'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def characterize():
    baseline = Path(__file__).resolve().parents[2]/"baselines/DeepEye-SQL"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    LLM = _frozen_module('llm', 'baselines/DeepEye-SQL/app/llm/llm.py').LLM
    from app.config.config import LLMConfig
    LLMExtractor = _frozen_module('extractor', 'baselines/DeepEye-SQL/app/llm_extractor/extractor.py').LLMExtractor
    from app.logger import logger
    from openai import APIConnectionError
    from openai._base_client import httpx2 as httpx
    from openai.types.chat import ChatCompletion
    from tenacity import stop_after_attempt, wait_none

    def case(script):
        calls = []
        def create(**kwargs):
            i = len(calls)
            kind = script[i] if i < len(script) else "valid"
            calls.append({"request": i, "response_id": f"r{i}" if kind != "error" else None,
                          "kind": kind, "n": kwargs["n"], "usage_total": 0 if kind=="error" else 30})
            if kind == "error":
                raise APIConnectionError(request=httpx.Request("POST", "http://fixture.invalid"))
            content = f"<result>r{i}</result>" if kind=="valid" else ("" if kind=="empty" else "malformed")
            return ChatCompletion.model_validate({"id": f"r{i}", "created": 1, "object": "chat.completion",
                "model": "fixture", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": content}}], "usage": {
                "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
        llm = LLM(LLMConfig(model="fixture", base_url="http://fixture.invalid", api_key="not-a-key",
                            n_call_strategy="split", max_tokens=16384, temperature=.6))
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        # Historical bounded_runner_factory budget; remove only its backoff sleep.
        llm.ask = MethodType(LLM.ask.retry_with(stop=stop_after_attempt(2), wait=wait_none()), llm)
        import re
        def parser(content):
            m = re.search(r"<result>(.*?)</result>", content)
            return m.group(1) if m else None
        with logger.contextualize(probe="legacy"):
            parsed, usage = LLMExtractor(max_retry=2).extract_with_retry(llm=llm,
                messages=[{"role": "user", "content": "fixture only"}], rule_parser=parser, n=5)
        return {"requests": len(calls), "retained_response_ids": parsed, "call_sequence": calls,
                "native_total_tokens": usage["total_tokens"], "reported_total_tokens": sum(c["usage_total"] for c in calls),
                "unknown_usage_requests": sum(c["kind"]=="error" for c in calls)}
    return {"source_revision": SOURCE_REVISION,
            "connection_then_success": case(["valid"]*3+["error"]),
            "parse_then_success": case(["valid"]*3+["malformed", "valid", "valid"]),
            "empty_then_success": case(["valid"]*3+["empty"]),
            "exhausted": case((["valid"]*3+["error"])*4)}
