import traceback
import time
from urllib.error import HTTPError

class Error(Exception):
    """Base class for other custom exceptions"""

    pass


class RetryAttemptsFailed(Error):
    """Raised when the retry attempts fail so that we can try to start again"""

    pass


def define_exception_handler_params(handler_type):
    """
    Defined parameters of retrying for known handler types
    :param handler_type: A type of handler
    :return: parameters for exception handling and retries
    """
    if handler_type == "yfinance":
        sleep_timer = 60
        retry_times = 60
        retry_exceptions = (HTTPError)
        terminal_exceptions = ()
    else:
        print(f"Incompatible handler_type: {handler_type}")
        exit(0)
    return sleep_timer, retry_times, retry_exceptions, terminal_exceptions


def exception_handler(handler_type):
    """
    Exceptions handler and retry decorator
    Retries the wrapped function/method `times` times if the exceptions are thrown
    :param handler_type: Type supported by define_exception_handler_params()
    """

    # Identify the retry/exception parameters
    (
        sleep_timer,
        retry_times,
        retry_exceptions,
        terminal_exceptions,
    ) = define_exception_handler_params(handler_type)

    # Run the retries decorator and return the result
    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            # Attempts per the parameters
            while attempt < retry_times:
                # log_object.info(f"Attempt iterations for the handler type '{handler_type}'")
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    print(
                        f"Known exception {e} thrown when attempting to run {func} [attempt {attempt+1} of {retry_times}]"
                    )
                    attempt += 1
                    time.sleep(sleep_timer)
                except terminal_exceptions as e:
                    err_msg = traceback.format_exc()
                    print(
                        f"Terminal error {e} occured\n" f"Full error message: {err_msg}"
                    )
                    exit(0)
                except KeyboardInterrupt:
                    print("KeyBoardInterrupt: exiting")
                    exit(0)
                except:
                    err_msg = traceback.format_exc()
                    print(
                        f"An unexpected error occurred [attempt {attempt+1} of {retry_times}]\n"
                        f"Full error message: {err_msg}"
                    )
                    attempt += 1
                    time.sleep(sleep_timer)
            # Raise an error if the exception is still happening
            if attempt >= retry_times:
                print("Retry attempts were unsuccessful")
                raise RetryAttemptsFailed()
            return func(*args, **kwargs)

        return newfn

    return decorator
