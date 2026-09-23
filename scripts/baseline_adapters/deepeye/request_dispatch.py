"""DeepEye policies and legacy API over the shared single-request core."""
from scripts.baseline_adapters.shared.transport import RequestLimits, RequestDispatcher as _SharedDispatcher
from .run_admission import AdaptiveAdmission


def _identity():
    from app.llm.sampling import sampling_identity
    return sampling_identity()


def _fatal(error):
    from app.llm.sampling import is_retryable_data_inspection_error
    return ((AdaptiveAdmission._status_code(error) in (400, 401, 403, 404, 422)
             and not is_retryable_data_inspection_error(error))
            or isinstance(error, (TypeError, ValueError)))


class RequestDispatcher(_SharedDispatcher):
    def __init__(self, limits=None, *, stop_event=None, emit=None):
        from app.llm.sampling import SamplingPaused
        super().__init__(limits, stop_event=stop_event, emit=emit,
                         identity_provider=_identity,
                         attempt_number=lambda identity: identity.get('sample_attempt', 1),
                         stopped_error=SamplingPaused, fatal_policy=_fatal)
