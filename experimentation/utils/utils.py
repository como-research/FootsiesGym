import functools


def timeout(seconds):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(
                f"Function {func.__name__} timed out after {seconds} seconds"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # TODO(chase): Figure out how to handle timeouts in Windows.
            # signal.signal(signal.SIGABRT, _handle_timeout)
            # signal.alarm(seconds)
            # try:
            #     result = func(*args, **kwargs)
            # finally:
            #     signal.alarm(0)
            return result

        return wrapper

    return decorator
