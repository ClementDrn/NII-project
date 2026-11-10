import argparse

import pytest

from scripts.vlm import send_vlm_request as model_request
from digit.generator.model_client import ModelType
from digit.generator.request import ImageBlock, TextBlock
from tests.fakes import FakeModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["combined", "sequential", "concurrent"])
async def test_cli_builds_current_requests_and_always_releases_client(
    tmp_path, monkeypatch, mode
):
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    client = FakeModelClient(["first", "second"])
    monkeypatch.setattr(model_request, "create_configured_client", lambda args: client)
    args = argparse.Namespace(
        image=images, system="system", prompt="Describe", concurrency=2
    )
    results = await model_request.send_requests(args, mode)
    assert results == (["first"] if mode == "combined" else ["first", "second"])
    assert all(request.system_text == "system" for request in client.requests)
    assert [
        block.image_path
        for request in client.requests
        for block in request.user_content
        if isinstance(block, ImageBlock)
    ] == images
    assert all(
        request.user_content[-1] == TextBlock("Describe") for request in client.requests
    )
    assert client.stop_calls == 1


@pytest.mark.asyncio
async def test_provider_failure_releases_cli_client(monkeypatch):
    client = FakeModelClient([ValueError("provider failed")])
    monkeypatch.setattr(model_request, "create_configured_client", lambda args: client)
    args = argparse.Namespace(image=[], system="", prompt="Question")
    with pytest.raises(ValueError, match="provider failed"):
        await model_request.send_requests(args, "combined")
    assert client.stop_calls == 1


@pytest.mark.asyncio
async def test_images_are_rejected_for_text_only_models(tmp_path, monkeypatch):
    client = FakeModelClient(model_type=ModelType.LLM)
    monkeypatch.setattr(model_request, "create_configured_client", lambda args: client)
    args = argparse.Namespace(
        image=[tmp_path / "image.png"], system="", prompt="Question"
    )
    with pytest.raises(ValueError, match="VLM"):
        await model_request.send_requests(args, "combined")
    assert client.start_calls == 0
