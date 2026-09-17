import asyncio

import anthropic

from untitled.core.base import CONFIG, LOG
from untitled.core.settings import get_host_config_by_id


# --- Batch configuration -----------------------------------------------------

BATCH_IDS = [
    "msgbatch_XXXXXXXX",
]

async def _print_requests(client: anthropic.AsyncAnthropic, batch_id: str) -> None:
    """Print the request results and errors for a given batch."""
    try:
        async for result in await client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                LOG.error(
                    "Failed request: batch=%s custom_id=%s type=%s details=%s",
                    batch_id,
                    result.custom_id,
                    result.result.type,
                    result.result,
                )
            # else:
            #     generated_text = "\n".join(
            #         block.text
            #         for block in result.result.message.content
            #         if block.type == "text"
            #     )
            #     LOG.info(
            #         "Successful request: batch=%s custom_id=%s instructions=\n%s",
            #         batch_id,
            #         result.custom_id,
            #         generated_text,
            #     )
    except Exception:
        LOG.exception("Could not inspect results for batch %s", batch_id)

async def main() -> int:
    # Load the Anthropic API key from the configured host.
    anthropic_host = get_host_config_by_id(CONFIG, "anthropic")
    client = anthropic.AsyncAnthropic(api_key=anthropic_host.api_key)

    try:
        # Retrieve and print the current state of every batch.
        for batch_id in BATCH_IDS:
            try:
                batch = await client.messages.batches.retrieve(batch_id)
                LOG.info(
                    "Batch %s: status=%s, request_counts=%s",
                    batch.id,
                    batch.processing_status,
                    batch.request_counts,
                )
                await _print_requests(client, batch_id)
            except Exception:
                LOG.exception("Could not retrieve batch %s", batch_id)
    finally:
        await client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
