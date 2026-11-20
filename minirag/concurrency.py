import asyncio
from functools import wraps


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Limit the number of concurrent calls to an async function."""

    def final_decro(func):
        current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal current_size
            while current_size >= max_size:
                await asyncio.sleep(waitting_time)
            current_size += 1
            try:
                return await func(*args, **kwargs)
            finally:
                current_size -= 1

        return wait_func

    return final_decro
