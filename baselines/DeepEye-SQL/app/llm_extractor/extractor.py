"""Rule parsing shares the sample's bounded transport retry budget."""
from app.llm.sampling import execute_group, MAX_SAMPLE_ATTEMPTS
from app.logger import logger

DEFAULT_LLM_EXTRACTOR_MAX_RETRY = 3


class LLMExtractor:
    def __init__(self, max_retry=DEFAULT_LLM_EXTRACTOR_MAX_RETRY):
        self._max_retry = max_retry

    @property
    def max_retry(self):
        """Legacy retries after the first attempt; evaluation sets an explicit total."""
        return self._max_retry

    def extract_with_retry(self, llm, messages, rule_parser, parser_kwargs=None,
                           fix_end_token=False, end_token='</result>', n=1,
                           max_retry=None, **llm_kwargs):
        retries = self.max_retry if max_retry is None else max_retry
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError('max_retry must be a nonnegative integer')
        def parse(message):
            content = message.content.strip()
            if fix_end_token and not content.endswith(end_token):
                content += end_token
            return rule_parser(content, **(parser_kwargs or {}))
        outcome = execute_group(lambda: llm.request_once(messages, **llm_kwargs), parse,
            n=n, max_attempts=getattr(llm, 'sample_max_attempts', min(retries + 1, MAX_SAMPLE_ATTEMPTS)))
        if not outcome.complete:
            logger.warning(f'Incomplete sampling group: {len(outcome.results)}/{n}')
        return outcome.results, outcome.effective_usage


def get_extractor(max_retry=DEFAULT_LLM_EXTRACTOR_MAX_RETRY):
    return LLMExtractor(max_retry=max_retry)
