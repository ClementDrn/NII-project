import asyncio

import pytest
import pytest_asyncio

from digit.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from digit.generator.request import Request, TextBlock
from tests.fakes import FakeModelClient


@pytest_asyncio.fixture
async def dispatcher_factory():
    dispatchers = []

    async def create(clients=()):
        for client in clients:
            await client.open_connection()
        dispatcher = PromptDispatcher(list(clients))
        dispatchers.append(dispatcher)
        await dispatcher.start()
        return dispatcher

    yield create
    for dispatcher in dispatchers:
        await asyncio.wait_for(dispatcher.stop(), timeout=2)


async def enqueue(dispatcher, text='user'):
    future = asyncio.get_running_loop().create_future()
    request = Request('system', [TextBlock(text)])
    await dispatcher.add(PromptRequest(request, future))
    return request, future


@pytest.mark.asyncio
async def test_start_and_stop_are_idempotent(dispatcher_factory):
    dispatcher = await dispatcher_factory()
    await dispatcher.start()
    assert await dispatcher.is_running()
    await dispatcher.stop()
    await dispatcher.stop()
    assert not await dispatcher.is_running()


@pytest.mark.asyncio
async def test_client_registration_and_removal(dispatcher_factory):
    first, second = FakeModelClient(), FakeModelClient()
    dispatcher = await dispatcher_factory([first])
    await dispatcher.add_clients([second])
    assert await dispatcher.get_clients() == [first, second]
    await dispatcher.remove_clients([first])
    assert await dispatcher.get_clients() == [second]


@pytest.mark.asyncio
async def test_processes_request_and_drains_queue(dispatcher_factory):
    client = FakeModelClient(['ok'])
    dispatcher = await dispatcher_factory([client])
    request, future = await enqueue(dispatcher)
    assert await asyncio.wait_for(future, timeout=2) == 'ok'
    await asyncio.wait_for(dispatcher.wait_until_all_prompts_processed(), timeout=2)
    assert client.requests == [request]


@pytest.mark.asyncio
async def test_waiting_prompt_is_served_by_late_client(dispatcher_factory):
    dispatcher = await dispatcher_factory()
    request, future = await enqueue(dispatcher)
    client = FakeModelClient(['late response'])
    await client.open_connection()
    await dispatcher.add_clients([client])
    assert await asyncio.wait_for(future, timeout=2) == 'late response'
    assert client.requests == [request]


@pytest.mark.asyncio
async def test_lowest_priority_value_is_selected_first(dispatcher_factory):
    low = FakeModelClient(['low'], priority=10)
    high = FakeModelClient(['high'], priority=0)
    dispatcher = await dispatcher_factory([low, high])
    _, future = await enqueue(dispatcher)
    assert await asyncio.wait_for(future, timeout=2) == 'high'
    assert low.requests == []


@pytest.mark.asyncio
async def test_provider_failure_does_not_lose_client_or_queue_task(dispatcher_factory):
    client = FakeModelClient([ValueError('bad request'), 'recovered'])
    dispatcher = await dispatcher_factory([client])
    _, failed = await enqueue(dispatcher)
    with pytest.raises(ValueError, match='bad request'):
        await asyncio.wait_for(failed, timeout=2)
    _, succeeded = await enqueue(dispatcher)
    assert await asyncio.wait_for(succeeded, timeout=2) == 'recovered'
    await asyncio.wait_for(dispatcher.wait_until_all_prompts_processed(), timeout=2)


@pytest.mark.asyncio
async def test_batch_size_allows_concurrent_requests(dispatcher_factory, monkeypatch):
    client = FakeModelClient(batch_size=2)
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = []

    async def prompt(request):
        started.append(request)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return request.user_content[0].text

    monkeypatch.setattr(client, 'prompt', prompt)
    dispatcher = await dispatcher_factory([client])
    _, first = await enqueue(dispatcher, 'first')
    _, second = await enqueue(dispatcher, 'second')
    try:
        await asyncio.wait_for(both_started.wait(), timeout=2)
        assert not first.done() and not second.done()
    finally:
        release.set()
    assert await asyncio.wait_for(asyncio.gather(first, second), timeout=2) == ['first', 'second']
