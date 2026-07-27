from untitled.core.base import LOG, CONFIG
from untitled.core.settings import GeneratorStepConfig, get_generator_config_by_id, get_model_instance_config_by_id
from untitled.generator.instruction_generator import get_api_parameters_from_config
from untitled.generator.batch import Batch, BatchStatus, BatchRequest, BatchRequestResult
from untitled.generator.model import create_model_client
from untitled.generator.request import TextBlock

import asyncio


# CHANGE GENERATOR AND MODEL INSTANCE IDS HERE, SHOULD REFER TO IDS IN config.yaml
GENERATOR_ID = "claude-opus-4.6_single_step"
MODEL_INSTANCE_ID = "claude-opus-4.6_0"


async def main() -> int:
    # --- Setup VLM ---
    LOG.debug("Setting up VLM interface...")
    # Fetch settings from config based on above constants
    generator_config = get_generator_config_by_id(CONFIG, GENERATOR_ID)
    model_config = get_model_instance_config_by_id(CONFIG, MODEL_INSTANCE_ID)
    
    # Find planning step config
    planning_step_config: GeneratorStepConfig
    for step in generator_config.steps:
        if step.id == "planning":
            planning_step_config = step
            break
    else:
        LOG.error("Step 'planning' could not be found in generator...")
        return 1

    # Enable model instance
    LOG.debug("Enabling model instance...")
    api_params = get_api_parameters_from_config(planning_step_config)
    vlm = create_model_client(model_config, api_params)
    if vlm is None:
        LOG.error("Failed to create model client.")
        return 1

    # Open connection
    LOG.debug("Opening connection to VLM...")
    await vlm.open_connection()

    # --- Create and send batch object ---
    # Create object
    LOG.debug("Creating and sending batch object...")
    batch: Batch = await vlm.batch(
        requests=[
            BatchRequest(request_id="id0", system_text="Repeat three times the following:", user_content=[TextBlock(text="Hello")]),
            BatchRequest(request_id="id1", system_text="Repeat three times the following:", user_content=[TextBlock(text="Bye")]),
        ]
    )

    # Send the batch
    batch_status: BatchStatus = await batch.send()
    LOG.debug("Batch was sent.")

    # --- Retrieve batch results ---
    # Wait for batch completion
    LOG.debug("Waiting for batch completion...")
    while batch_status not in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED):   
        batch_status = await batch.check_status()
        # Avoid blocking the event loop while polling the batch state.
        await asyncio.sleep(15)

    LOG.debug(f"Batch finished with status: {batch_status}")

    if batch_status == BatchStatus.COMPLETED:
        # Download and read results
        result: BatchRequestResult
        async for result in batch.download_results():
            LOG.info(f"Result for request {result.request_id}: Status: {result.status}, Text: {result.text}, Thinking: {result.thinking}")

    # Close the connection
    LOG.debug("Closing connection to VLM...")
    await vlm.close_connection()

    LOG.debug("Done.")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
