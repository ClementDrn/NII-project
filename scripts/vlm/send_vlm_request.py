"""Send explicit text/image requests through a configured model instance."""

import argparse
import asyncio
import json
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-instance", required=True, help="Model instance ID from config.yaml"
    )
    parser.add_argument("--max-tokens", type=positive_int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--thinking", choices=["default", "enabled", "disabled"], default="default"
    )


def create_configured_client(args):
    from digit.core.base import CONFIG
    from digit.core.settings import GeneratorStepConfig, get_model_instance_config_by_id
    from digit.generator.instruction_generator import get_api_parameters_from_config
    from digit.generator.model import create_model_client

    instance = get_model_instance_config_by_id(CONFIG, args.model_instance)
    step = GeneratorStepConfig(
        id="request",
        model_name=instance.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking_mode=args.thinking,
    )
    client = create_model_client(instance, get_api_parameters_from_config(step))
    if client is None:
        raise ValueError(
            f"Model instance {instance.id!r} must define one local or remote configuration"
        )
    return client


async def send_requests(args, mode: str):
    from digit.generator.model_client import ModelType
    from digit.generator.request import Request, TextBlock, ImageBlock

    client = create_configured_client(args)
    if args.image and client.model_type != ModelType.VLM:
        raise ValueError("Image requests require a VLM model instance")
    images_per_request = (
        [args.image] if mode == "combined" else [[image] for image in args.image]
    )
    requests = [
        Request(
            args.system,
            [*(ImageBlock(image) for image in images), TextBlock(args.prompt)],
        )
        for images in images_per_request
    ]
    try:
        await client.open_connection()
        if mode == "concurrent":
            semaphore = asyncio.Semaphore(args.concurrency)

            async def send(request):
                async with semaphore:
                    return await client.prompt(request)

            return await asyncio.gather(*(send(request) for request in requests))
        return [await client.prompt(request) for request in requests]
    finally:
        await client.stop()


def main(mode="combined", require_images=False) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Send all images in one request"
            if mode == "combined"
            else f"Send one request per image ({mode})"
        )
    )
    add_model_arguments(parser)
    parser.add_argument("--prompt", required=True, help="User prompt text")
    parser.add_argument("--system", default="", help="System prompt text")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        required=require_images,
        help="Image path; repeat for multiple images",
    )
    if mode == "concurrent":
        parser.add_argument("--concurrency", type=positive_int, default=4)
    args = parser.parse_args()
    for image in args.image:
        if not image.is_file():
            parser.error(f"Image does not exist: {image}")
    from digit.core.base import initialize_runtime

    initialize_runtime()
    responses = asyncio.run(send_requests(args, mode))
    print(
        responses[0]
        if mode == "combined"
        else json.dumps(responses, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(mode="combined", require_images=True))
