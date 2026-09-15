"""Offline acceptance fixtures: real LLM/extractor, only the SDK is replaced."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError
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


def inspection_error(*, message='Output data may contain inappropriate content.',
                     code='data_inspection_failed', status=400, nested=False):
    body = {'code': code, 'type': code, 'message': message, 'param': None}
    return BadRequestError(f'Error code: {status} - {body}', response=httpx.Response(status,
        request=httpx.Request('POST', 'https://invalid.test')),
        body={'error': body} if nested else body)


class SamplingTests(unittest.TestCase):
    def test_output_inspection_timeout_and_parser_share_four_attempts(self):
        timeout = APITimeoutError(request=httpx.Request('POST', 'https://invalid.test'))
        for nested in (False, True):
            with self.subTest(nested=nested):
                result, usage, calls, events = self.collect(
                    [timeout, inspection_error(nested=nested), response('bad'), response()], n=1)
                self.assertEqual((len(result), len(calls), usage['total_tokens']), (1, 4, 30))
                attempts = [p for k, p in events if k == 'sample_attempt']
                self.assertEqual([p['sample_attempt'] for p in attempts], [1, 2, 3, 4])
                self.assertEqual([p['status'] for p in attempts],
                                 ['api_error', 'api_error', 'parse_rejected', 'succeeded'])
                self.assertIn('APITimeoutError', attempts[0]['error'])
                self.assertIn('data_inspection_failed', attempts[1]['error'])
                self.assertIsNone(attempts[1]['usage'])
                self.assertTrue(all(call == calls[0] for call in calls))

    def test_exhausted_output_inspection_keeps_other_samples_without_extra_budget(self):
        result, usage, calls, events = self.collect(
            [response('SELECT kept')] + [inspection_error()] * 4 + [response('SELECT later')], n=3)
        self.assertEqual(result, ['SELECT kept', 'SELECT later'])
        self.assertEqual((len(calls), usage['total_tokens']), (6, 60))
        samples = [p for k, p in events if k == 'sample_result']
        self.assertEqual([p['attempt_count'] for p in samples], [1, 4, 1])
        self.assertFalse(samples[1]['succeeded'])
        self.assertFalse(samples[1]['fatal'])
        self.assertFalse(next(p for k, p in events if k == 'sampling_group_result')['complete'])

    def test_input_unknown_and_non400_inspection_errors_are_not_retried(self):
        errors = [inspection_error(message='Input data may contain inappropriate content.'),
                  inspection_error(message='Data may contain inappropriate content.'),
                  inspection_error(code='invalid_parameter'),
                  inspection_error(status=401), inspection_error(status=403),
                  inspection_error(status=422)]
        for error in errors:
            with self.subTest(body=error.body, status=error.status_code):
                result, usage, calls, events = self.collect([error, response()], n=1)
                self.assertEqual((result, len(calls), usage['total_tokens']), ([], 1, 0))
                self.assertTrue(next(p for k, p in events if k == 'sample_result')['fatal'])

    def test_callable_instance_and_partial_over_instance_are_rejected_before_transport(self):
        from functools import partial
        from app.llm.sampling import SamplingIdentityError
        class Rule:
            def __init__(self):
                self.suffix = ' configured'
            def __call__(self, content):
                return content + self.suffix
        for rule in (Rule(), partial(Rule())):
            with self.subTest(rule=rule):
                llm, requests = llm_fixture([response()])
                with self.assertRaisesRegex(SamplingIdentityError, 'unsupported parser callable'):
                    LLMExtractor().extract_with_retry(llm, [], rule, n=1)
                self.assertEqual(requests, [])

    def test_unsupported_native_callable_is_rejected_before_transport(self):
        from app.llm.sampling import SamplingIdentityError
        llm, requests = llm_fixture([response()])
        with self.assertRaisesRegex(SamplingIdentityError, 'unsupported parser callable'):
            LLMExtractor().extract_with_retry(llm, [], str.strip, n=1)
        self.assertEqual(requests, [])

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
        with observe_sampling(lambda kind, payload: events.append((kind, payload))), patch('app.llm_extractor.extractor.logger.warning') as warning:
            result, usage = LLMExtractor().extract_with_retry(llm, [], parse, n=n, **kwargs)
        self.assertEqual(warning.call_count, int(len(result) < n))
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
