from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from openai import OpenAI, AzureOpenAI
from .sampling import execute_group, SamplingIncompleteError, MAX_SAMPLE_ATTEMPTS

if TYPE_CHECKING:
    from app.config.config import LLMConfig


class EmptyResponseError(Exception):
    """Import-compatible exception for native callers."""


class LLM:
    """One client per wrapper; one SDK invocation per request_once."""
    def __init__(self, llm_config: LLMConfig):
        self._config = llm_config
        self._client = None
        self._client_lock = threading.Lock()
        self.sample_max_attempts = MAX_SAMPLE_ATTEMPTS

    @property
    def llm_config(self):
        return self._config

    def _create_client(self):
        params = dict(api_key=self._config.api_key, base_url=self._config.base_url, max_retries=0)
        if self._config.api_type == 'openai':
            return OpenAI(**params)
        if self._config.api_type == 'azure':
            return AzureOpenAI(**params, api_version=self._config.api_version)
        raise ValueError(f'Unsupported api type: {self._config.api_type}')

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def request_once(self, messages, system_message=None, timeout=300, **kwargs):
        """Single transport operation without content/parser retries or repair."""
        params = dict(model=self._config.model,
            messages=[system_message] + messages if system_message else messages,
            max_tokens=self._config.max_tokens, temperature=self._config.temperature,
            timeout=timeout)
        if self._config.reasoning_effort is not None:
            params['reasoning_effort'] = self._config.reasoning_effort
        if self._config.extra_body:
            params['extra_body'] = self._config.extra_body
        params.update(kwargs)
        params['n'] = 1
        return self._get_client().chat.completions.create(**params)

    def ask(self, messages, system_message=None, timeout=None, **kwargs):
        n = kwargs.pop('n', 1)
        if timeout is not None:
            kwargs['timeout'] = timeout
        outcome = execute_group(lambda: self.request_once(messages,
            system_message=system_message, **kwargs), lambda message: message,
            n=n, max_attempts=self.sample_max_attempts,
            recovery_identity=self.sampling_request_identity(messages, system_message=system_message,
                                                              parser={'kind': 'message'}, **kwargs))
        if not outcome.complete:
            raise SamplingIncompleteError(outcome)
        return outcome.results, outcome.effective_usage

    def sampling_request_identity(self, messages, system_message=None, *, parser=None, **kwargs):
        """Exact logical request, excluding credentials; shared by both callers."""
        config = self._config
        params = dict(model=config.model, max_tokens=config.max_tokens,
                      temperature=config.temperature, reasoning_effort=config.reasoning_effort,
                      extra_body=config.extra_body, api_type=config.api_type,
                      base_url=str(config.base_url), api_version=getattr(config, 'api_version', None))
        params.update(kwargs)
        return {'messages': [system_message] + messages if system_message else messages,
                'params': params, 'parser': parser}
