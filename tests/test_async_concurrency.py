import asyncio
import importlib.util
import unittest
from pathlib import Path


# Load the standard-library-only helper directly so this unit test does not
# initialize MiniRAG or its model/database dependencies.
_CONCURRENCY_PATH = Path(__file__).resolve().parents[1] / "minirag" / "concurrency.py"
_SPEC = importlib.util.spec_from_file_location(
    "minirag_concurrency_test", _CONCURRENCY_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load concurrency helper from {_CONCURRENCY_PATH}")
_CONCURRENCY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONCURRENCY)
limit_async_func_call = _CONCURRENCY.limit_async_func_call


class LimitAsyncFuncCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_releases_slot_when_wrapped_call_raises(self):
        attempts = 0

        @limit_async_func_call(max_size=1, waitting_time=0)
        async def synthetic_provider():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("synthetic provider failure")
            return "recovered"

        with self.assertRaisesRegex(RuntimeError, "synthetic provider failure"):
            await synthetic_provider()

        result = await asyncio.wait_for(synthetic_provider(), timeout=0.25)
        self.assertEqual(result, "recovered")
        self.assertEqual(attempts, 2)


if __name__ == "__main__":
    unittest.main()
