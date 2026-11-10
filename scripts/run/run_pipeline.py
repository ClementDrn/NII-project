"""Run DIGIT jobs with cancellation-safe cleanup."""

import argparse
import asyncio


async def run_pipeline(pipeline):
    """Release initialized components on success, error, or task cancellation."""
    try:
        await pipeline.init()
        await pipeline.start()
    finally:
        await pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the generation and simulation pipeline"
    )
    parser.add_argument(
        "--jobs-filepath", default="jobs.yaml", help="Path to jobs configuration"
    )
    args = parser.parse_args()
    from digit.core.base import LOG, initialize_runtime
    from digit.core.pipeline import Pipeline

    initialize_runtime()
    try:
        # asyncio.run cancels the main task on the first Ctrl+C, allowing the
        # finally block to finish. A second Ctrl+C interrupts cleanup.
        asyncio.run(run_pipeline(Pipeline(jobs_filepath=args.jobs_filepath)))
    except KeyboardInterrupt:
        LOG.info("Pipeline interrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
