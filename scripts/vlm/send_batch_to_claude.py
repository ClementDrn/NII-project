"""Submit explicit JSONL requests through the native Anthropic batch API."""

import argparse
import asyncio
import json
from pathlib import Path

from send_vlm_request import (
    add_model_arguments,
    create_configured_client,
    positive_int,
)


def load_requests(path):
    from digit.generator.batch import BatchRequest
    from digit.generator.request import TextBlock

    requests = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        requests.append(
            BatchRequest(
                str(item["id"]), item.get("system", ""), [TextBlock(item["prompt"])]
            )
        )
    if not requests:
        raise ValueError("Request file is empty")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("Request IDs must be unique")
    return requests


async def submit_batch(args):
    from digit.generator.batch import BatchStatus
    from digit.generator.providers.anthropic import AnthropicAdapter

    requests = load_requests(args.requests)
    client = create_configured_client(args)
    if not isinstance(client.adapter, AnthropicAdapter):
        raise ValueError("This command requires an Anthropic model instance")
    try:
        await client.open_connection()
        batch = await client.batch(requests)
        status = await batch.send()
        print(
            json.dumps({"batch_id": batch.get_id(), "status": status.value}), flush=True
        )
        if not args.wait:
            return 0
        async with asyncio.timeout(args.timeout):
            while status not in {
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.CANCELLED,
                BatchStatus.EXPIRED,
            }:
                await asyncio.sleep(args.poll_interval)
                status = await batch.check_status()
            if status != BatchStatus.COMPLETED:
                return 1
            async for result in batch.download_results():
                print(
                    json.dumps(
                        {
                            "id": result.request_id,
                            "status": result.status.value,
                            "text": result.text,
                            "thinking": result.thinking,
                        }
                    )
                )
        return 0
    finally:
        await client.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument(
        "requests", type=Path, help="JSONL with id, prompt and optional system fields"
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for completion and print results"
    )
    parser.add_argument("--poll-interval", type=positive_int, default=15)
    parser.add_argument(
        "--timeout", type=positive_int, default=86400, help="Maximum wait in seconds"
    )
    args = parser.parse_args()
    if not args.requests.is_file():
        parser.error(f"Request file does not exist: {args.requests}")
    from digit.core.base import initialize_runtime

    initialize_runtime()
    return asyncio.run(submit_batch(args))


if __name__ == "__main__":
    raise SystemExit(main())
