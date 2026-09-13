"""Offline acceptance fixtures: real LLM/extractor, only the SDK is replaced."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, AuthenticationError, BadRequestError
from openai.types.chat import ChatCompletion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines/DeepEye-SQL'))
from app.llm import LLM
from app.llm_extractor import LLMExtractor


def response(content='SELECT wrong_but_parseable', tokens=30, reasoning=None):
    usage = None if tokens is None else dict(prompt_tokens=10 if tokens == 30 else 3,
        completion_tokens=20 if tokens == 30 else 4, total_tokens=tokens,
        completion_tokens_details=dict(reasoning_tokens=reasoning))
    return ChatCompletion.model_validate(dict(id='fixture', created=1, object='chat.completion',
        model='fixture', choices=[dict(index=0, finish_reason='stop',
            message=dict(role='assistant', content=content))], usage=usage))


def llm_fixture(sequence):
    requests = []
    pending = iter(sequence)
    def create(**kwargs):
        requests.append(kwargs)
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        return value
    config = NS(model='fixture', temperature=0.6, reasoning_effort=None,
        max_tokens=16384, n_call_strategy='split', max_request_n=1,
        extra_body={}, api_type='openai', api_key='offline', base_url='https://invalid.test')
    llm = LLM(config)
    llm._client = NS(chat=NS(completions=NS(create=create)), close=lambda: None)
    return llm, requests


def parse(content):
    return content if content.startswith('SELECT') else None


class SamplingTests(unittest.TestCase):
    def test_five_samples_one_parse_rejection_costs_six_calls_and_150_effective(self):
        llm, requests = llm_fixture([response()] * 3 + [response('invalid', 7)] + [response()] * 2)
        results, usage = LLMExtractor().extract_with_retry(llm, [], parse, n=5)
        self.assertEqual(len(results), 5)  # duplicate and wrong SQL still count
        self.assertEqual(len(requests), 6)
        self.assertEqual(usage['total_tokens'], 150)
        self.assertTrue(all(call['n'] == 1 for call in requests))

    def collect(self, sequence, n=5, **kwargs):
        from app.llm.sampling import observe_sampling
        events = []
        llm, requests = llm_fixture(sequence)
        with observe_sampling(lambda kind, payload: events.append((kind, payload))):
            result, usage = LLMExtractor().extract_with_retry(llm, [], parse, n=n, **kwargs)
        return result, usage, requests, events

    def test_fixed_sample_identity_success_is_not_repeated(self):
        results, usage, calls, events = self.collect(
            [response()] * 3 + [response('invalid', 7)] + [response()] * 2)
        attempts = [p for k, p in events if k == 'sample_attempt']
        self.assertEqual(Counter(p['sample_index'] for p in attempts), {0: 1, 1: 1, 2: 1, 3: 2, 4: 1})
        self.assertEqual([p for k, p in events if k == 'sampling_group_result'][0]['success_count'], 5)
        self.assertEqual(sum(p['usage']['total_tokens'] for p in attempts), 157)

    def test_four_failed_attempts_leave_four_of_five_not_complete(self):
        error = APIConnectionError(request=httpx.Request('POST', 'https://invalid.test'))
        results, usage, calls, events = self.collect([response()] * 3 + [error] * 4 + [response()])
        self.assertEqual((len(results), len(calls), usage['total_tokens']), (4, 8, 120))
        group = [p for k, p in events if k == 'sampling_group_result'][0]
        self.assertFalse(group['complete'])
        self.assertEqual(group['success_count'], 4)
        self.assertEqual(sum(p['usage'] is None for k, p in events if k == 'sample_attempt'), 4)

    def test_api_empty_and_parse_rejection_share_four_total_attempts(self):
        error = APIConnectionError(request=httpx.Request('POST', 'https://invalid.test'))
        results, usage, calls, events = self.collect([error, response(''), response('bad'), response()], n=1)
        self.assertEqual((len(results), len(calls), usage['total_tokens']), (1, 4, 30))
        self.assertEqual([p['status'] for k, p in events if k == 'sample_attempt'],
                         ['api_error', 'empty', 'parse_rejected', 'succeeded'])

    def test_fourth_attempt_succeeds_without_fifth(self):
        results, usage, calls, events = self.collect([response()] * 3 + [response('bad')] * 3 + [response()] * 2)
        self.assertEqual((len(results), len(calls), usage['total_tokens']), (5, 8, 150))

    def test_deterministic_errors_stop_group_and_retain_prior_success(self):
        for error_type in (AuthenticationError, BadRequestError):
            error = error_type('context tokens invalid', response=httpx.Response(400,
                request=httpx.Request('POST', 'https://invalid.test')), body=None)
            result, usage, calls, events = self.collect([response(), error])
            self.assertEqual((len(result), len(calls), usage['total_tokens']), (1, 2, 30))
            self.assertEqual(calls[-1]['max_tokens'], 16384)

    def test_success_without_usage_is_retained_and_reasoning_is_not_added(self):
        result, usage, calls, events = self.collect([response(tokens=None), response(reasoning=12)], n=2)
        self.assertEqual((len(result), len(calls), usage['total_tokens']), (2, 2, 30))
        samples = [p for k, p in events if k == 'sample_result']
        self.assertIsNone(samples[0]['usage'])
        self.assertEqual(samples[1]['usage']['reasoning_tokens'], 12)

    def test_direct_ask_shares_sample_retry_algorithm(self):
        llm, calls = llm_fixture([response(), response(''), response()])
        messages, usage = llm.ask([], n=2)
        self.assertEqual((len(messages), len(calls), usage['total_tokens']), (2, 3, 60))

    def test_client_creation_disables_sdk_retries(self):
        llm, _ = llm_fixture([])
        with patch('app.llm.llm.OpenAI') as client:
            llm._create_client()
        self.assertEqual(client.call_args.kwargs.get('max_retries'), 0)

    def test_local_failure_is_not_retried_as_transient_api_error(self):
        result, usage, calls, events = self.collect([RuntimeError('local persistence failure')])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, [])
