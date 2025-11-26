from untitled.core.base import LOG
from untitled.core.pipeline import Pipeline
from untitled.generator.vlm import VLM
from untitled.simulator.simulator import Simulator
from untitled.simulator.simulation_data import SimulationData

import signal
import sys


### MAIN FUNCTION #############################################################
def main():
    # Instatiate pipeline
    pipeline = Pipeline()

    # Define interruption signal handler for graceful shutdown of pipeline
    def sigint_handler(sig, frame):
        LOG.info("Caught SIGINT, exiting gracefully...")
        pipeline.stop()
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, sigint_handler)

    LOG.info("Initializing pipeline...")
    pipeline.init()

    LOG.info("Starting pipeline...")
    pipeline.start()

    LOG.info("Stopping pipeline...")
    pipeline.stop()


if __name__ == "__main__":
    main()
