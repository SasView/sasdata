import warnings

# TODO: Anything deprecated should have that in the docstring!!!!! -- @pkienzle


def deprecated(replaced_with=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def decorator(func):
        def new_func(*args, **kwargs):
            warnings.simplefilter('once', DeprecationWarning)  # turn off filter
            warnings.warn(f"Call to deprecated function {func.__name__}. Call {replaced_with} in the future.",
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
    return decorator
