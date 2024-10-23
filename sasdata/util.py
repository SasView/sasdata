from typing import TypeVar, Callable

T = TypeVar("T")

def cache[T](fun: Callable[[], T]):
    """ Decorator to store values """

    cache_state = [False, None]

    def wrapper() -> T:
        if not cache_state[0]:
            cache_state[0] = True
            cache_state[1] = fun()

        return cache_state[1]

    return wrapper