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
async def test_prompt_dispatcher_init_and_getters():
    # Create clients and dispatcher
    client1 = FakeClient("c1", "ok")
    client2 = FakeClient("c2", "ok")
    dispatcher = PromptDispatcher([client1, client2])

    # Check that clients are registered
    clients = await dispatcher.get_clients()
    assert len(clients) == 2
    assert client1 in clients
    assert client2 in clients

@pytest.mark.asyncio
async def test_prompt_dispatcher_add_and_remove_clients():
    # Initial dispatcher with one client
    client1 = FakeClient("c1", "ok")
    dispatcher = PromptDispatcher([client1])

    # Add clients
    client2 = FakeClient("c2", "ok")
    client3 = FakeClient("c3", "ok")
    await dispatcher.add_clients([client2, client3])

    # Check that clients are added
    clients = await dispatcher.get_clients()
    assert len(clients) == 3
    assert client1 in clients
    assert client2 in clients
    assert client3 in clients

    # Remove a client
    await dispatcher.remove_clients([client2])

    # Check that client2 is removed
    clients = await dispatcher.get_clients()
    assert len(clients) == 2
    assert client1 in clients
    assert client2 not in clients
    assert client3 in clients

@pytest.mark.asyncio
async def test_prompt_dispatcher_start_and_stop():
    client = FakeClient("c1", "ok")
    dispatcher = PromptDispatcher([client])

    # Start
    await dispatcher.start()
    assert await dispatcher.is_running()

    # Subsequent start calls should not raise
    await dispatcher.start()

    # Stop
    await dispatcher.stop()
    assert not await dispatcher.is_running()

    # Subsequent stop calls should not raise
    await dispatcher.stop()

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
async def test_prompt_dispatcher_processes_single_prompt_without_wait():
    loop = asyncio.get_running_loop()
    client = FakeClient("c1", "ok")
    dispatcher = PromptDispatcher([client])

    await dispatcher.start()

    future = loop.create_future()
    await dispatcher.add(PromptRequest("sys", "user", [], future))

    result = await future
    assert result == "c1:ok"
    assert client.calls == [("sys", "user")]

    await dispatcher.stop()

@pytest.mark.asyncio
async def test_prompt_dispatcher_processes_single_prompt_with_late_client():
    loop = asyncio.get_running_loop()
    dispatcher = PromptDispatcher()
    await dispatcher.start()

    # Add prompt to dispatcher
    future = loop.create_future()
    await dispatcher.add(PromptRequest("sys", "user", [], future))

    # Add client after prompt
    client = FakeClient("c1", "ok")
    await dispatcher.add_clients([client])

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
