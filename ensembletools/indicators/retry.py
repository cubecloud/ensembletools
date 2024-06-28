from time import sleep
import logging
import traceback
import functools

__version__ = 0.018

logger = logging.getLogger()


def retry(retry_num, delay):
    """
    retry help decorator.

    Args
        delay (float):
        retry_num(int): the retry num; retry sleep sec
    Returns
        decorator
    """

    def decorator(func):
        """decorator"""

        # preserve information about the original function, or the func name will be "wrapper" not "func"
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            for attempt in range(retry_num):
                try:
                    return func(*args, **kwargs)  # should return the raw function's return value
                except Exception as err:  # pylint: disable=broad-except
                    logger.error(err)
                    logger.error(traceback.format_exc())
                    sleep(delay)
                logger.error("Trying attempt %s of %s.", attempt + 1, retry_num)
            logger.error("func %s retry failed", func)
            raise Exception('Exceed max retry num: {} failed'.format(retry_num))

        return wrapper

    return decorator
