import time
from typing import Callable, Tuple, TypeVar, Awaitable

TimingsResponse = TypeVar("TimingsResponse")


def with_timings_sync(fn: Callable[..., TimingsResponse]) -> Tuple[float, TimingsResponse]:
    start_time: float = time.time() * 1000
    fn_response: TimingsResponse = fn()
    end_time: float = time.time() * 1000
    return round(end_time - start_time, 1), fn_response


async def with_timings_async(fn: Awaitable[TimingsResponse]) -> Tuple[float, TimingsResponse]:
    start_time: float = time.time() * 1000
    fn_response: TimingsResponse = await fn
    end_time: float = time.time() * 1000
    return round(end_time - start_time, 1), fn_response
