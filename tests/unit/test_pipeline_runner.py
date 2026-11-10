import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run/run_pipeline.py"
spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.asyncio
async def test_cancellation_always_stops_pipeline():
    started = asyncio.Event()
    pipeline = AsyncMock()

    async def running():
        started.set()
        await asyncio.Event().wait()

    pipeline.start.side_effect = running
    task = asyncio.create_task(module.run_pipeline(pipeline))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pipeline.stop.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_initialization_failure_still_cleans_up():
    pipeline = AsyncMock()
    pipeline.init.side_effect = ValueError("bad configuration")
    with pytest.raises(ValueError, match="bad configuration"):
        await module.run_pipeline(pipeline)
    pipeline.start.assert_not_awaited()
    pipeline.stop.assert_awaited_once_with()
