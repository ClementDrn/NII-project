from untitled.core.base import LOG
from untitled.core.pipeline import Pipeline

import asyncio
import signal
import sys


### MAIN FUNCTION #############################################################
async def main():
    # Instantiate pipeline
    pipeline = Pipeline()

    # Define interruption signal handler
    # It does not exit immediately on first SIGINT to allow graceful shutdown
    shutdown_requested = False
    def sigint_handler(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            LOG.info("Caught SIGINT, initiating graceful shutdown...")
            shutdown_requested = True
        else:
            LOG.warning("Force exit requested")
            sys.exit(1)

    # Register signal handler
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        LOG.info("Initializing pipeline...")
        await pipeline.init()

        LOG.info("Starting pipeline...")
        await pipeline.start()
    except KeyboardInterrupt:
        LOG.info("Pipeline interrupted")
    finally:    # Ensure pipeline is stopped
        LOG.info("Stopping pipeline...")
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
