import asyncio

import pytest

from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest


class FakeClient:
    def __init__(self, name: str, response: str, priority: int = 0, batch_size: int = 1):
        self.name = name
        self.response = response
        self.priority = priority
        self.batch_size = batch_size
        self.calls: list[tuple[str, str]] = []

    async def prompt(self, system_text: str, user_text: str, user_image_paths):
        await asyncio.sleep(0)  # yield control to mimic async workload
        self.calls.append((system_text, user_text))
        return f"{self.name}:{self.response}"


@pytest.mark.asyncio
async def test_prompt_dispatcher_processes_single_prompt():
    loop = asyncio.get_running_loop()
    client = FakeClient("c1", "ok")
    dispatcher = PromptDispatcher([client])

    await dispatcher.start()

    future = loop.create_future()
    await dispatcher.add(PromptRequest("sys", "user", [], future))

    await dispatcher.wait_until_all_prompts_processed()
    result = await future
    assert result == "c1:ok"
    assert client.calls == [("sys", "user")]

    await dispatcher.stop()


@pytest.mark.asyncio
async def test_prompt_dispatcher_honours_extra_clients():
    loop = asyncio.get_running_loop()
    client_a = FakeClient("primary", "ok", priority=0)
    client_b = FakeClient("secondary", "ok", priority=1)
    dispatcher = PromptDispatcher([client_a])

    await dispatcher.start()

    futures = [loop.create_future() for _ in range(2)]
    await dispatcher.add(PromptRequest("sys1", "u1", [], futures[0]))
    await dispatcher.add(PromptRequest("sys2", "u2", [], futures[1]))

    await dispatcher.add_clients([client_b])
    await dispatcher.wait_until_all_prompts_processed()

    results = await asyncio.gather(*futures)
    assert sorted(results) == ["primary:ok", "secondary:ok"]
    assert len(client_a.calls) == 1
    assert len(client_b.calls) == 1

@pytest.mark.asyncio
async def test_prompt_dispatcher_shutdown_stops_thread():
    client = FakeClient("c1", "ok")
    dispatcher = PromptDispatcher([client])

    await dispatcher.start()
    await dispatcher.stop()

    # Subsequent stop calls should not raise
    await dispatcher.stop()
