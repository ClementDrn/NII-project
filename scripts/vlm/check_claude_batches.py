"""Inspect explicitly supplied Anthropic batch IDs."""

import argparse
import asyncio


async def inspect_batches(batch_ids, host_id):
    import anthropic
    from digit.core.base import CONFIG, LOG
    from digit.core.settings import get_host_config_by_id

    host = get_host_config_by_id(CONFIG, host_id)
    failed = False
    async with anthropic.AsyncAnthropic(api_key=host.api_key) as client:
        for batch_id in batch_ids:
            batch = await client.messages.batches.retrieve(batch_id)
            LOG.info(
                "Batch %s: status=%s counts=%s",
                batch.id,
                batch.processing_status,
                batch.request_counts,
            )
            if batch.processing_status != "ended":
                continue
            async for result in await client.messages.batches.results(batch_id):
                if result.result.type != "succeeded":
                    failed = True
                    LOG.error(
                        "Batch %s request %s: %s",
                        batch_id,
                        result.custom_id,
                        result.result.type,
                    )
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_ids", nargs="+")
    parser.add_argument(
        "--host-id", default="anthropic", help="Host ID from hosts.yaml"
    )
    args = parser.parse_args()
    from digit.core.base import initialize_runtime

    initialize_runtime()
    return asyncio.run(inspect_batches(args.batch_ids, args.host_id))


if __name__ == "__main__":
    raise SystemExit(main())
