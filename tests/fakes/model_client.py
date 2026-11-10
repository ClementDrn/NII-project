"""Scripted model boundary used with real tasks and prompt dispatchers."""
from digit.generator.model_client import ModelClient, ModelType
from digit.generator.request import Request


class FakeModelClient(ModelClient):
    def __init__(self, responses=(), *, parallel=True, model_type=ModelType.VLM,
                 priority=0, batch_size=1):
        super().__init__(model_type, priority, batch_size, adapter=None)
        self.responses = iter(responses)
        self.requests: list[Request] = []
        self.parallel = parallel
        self.start_calls = 0
        self.stop_calls = 0
        self.hot_calls = 0

    async def start(self):
        self.start_calls += 1
        self.state = self.State.STARTED

    async def open_connection(self):
        if self.state == self.State.DISABLED:
            await self.start()
        self.state = self.State.CONNECTED

    async def close_connection(self):
        self.state = self.State.STARTED

    async def stop(self):
        self.stop_calls += 1
        self.state = self.State.DISABLED

    async def set_hot(self):
        self.hot_calls += 1

    async def prompt(self, request: Request):
        if self.state != self.State.CONNECTED:
            raise ConnectionError("Fake model is not connected")
        self.requests.append(request)
        try:
            response = next(self.responses)
        except StopIteration:
            raise AssertionError("No scripted response for this request") from None
        if isinstance(response, Exception):
            raise response
        return response

    def is_remote(self):
        return True

    def is_parallelizable(self):
        return self.parallel
