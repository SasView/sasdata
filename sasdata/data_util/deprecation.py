import logging
import functools

# TODO: Anything deprecated should have that in the docstring!!!!! -- @pkienzle

logger = logging.getLogger(__name__)


def deprecated(replaced_with=None):
    """This is a decorator which can be used to mark functions as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps
    def decorator(func):
        def new_func(*args, **kwargs):
            logger.warning(f"Call to deprecated function {func.__name__}. Call {replaced_with} in the future.")
            return func(*args, **kwargs)
        return new_func
    return decorator
