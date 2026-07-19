from untitled.core.base import LOG, CONFIG
from untitled.generator.model_client import ModelClient, BatchResult, BatchStatus 
import time


# CHANGE GENERATOR AND MODEL INSTANCE IDS HERE, SHOULD REFER TO IDS IN config.yaml
GENERATOR_ID = "claude-opus-4.6_single_step"
MODEL_INSTANCE_ID = "claude-opus-4.6_0"


async def main() -> int:
    # --- Setup VLM ---
    print("Setting up VLM interface...")
    # Fetch settings from config based on above constants
    generator_config = get_generator_config_by_id(CONFIG, GENERATOR_ID)
    model_config = get_model_instance_config_by_id(CONFIG, MODEL_INSTANCE_ID)
    
    # Find planning step config
    planning_step_config: GeneratorStepConfig
    for step in generator_config.steps:
        if step.id == "planning":
            planning_step_config = step
            break
    if planning_step_config is None:
        LOG.error("Step 'planning' could not be found in generator...")
        return 1

    # Enable model instance with   
    api_params = get_api_parameters_from_config(planning_step_config)
    vlm = create_model_client(model_config, api_params)
    await vlm.start()
    print("Setup is done.")

    # --- Create and send batch object ---
    # Create object
    print("Creating and sending batch object...")
    batch_file = vlm.create_batch_file([
        { "id": "id0", "system_text": "Repeat three times the following:", "user_text": "Hello", "user_images": None },
        { "id": "id1", "system_text": "Repeat three times the following:", "user_text": "Bye", "user_images": None },
    ])
    print("Created batch file.")
    # Send the batch
    batch_response = vlm.send_batch(batch_file)
    print("Batch was sent.")

    # --- Retrieve batch results ---
    # Wait for batch completion
    print("Waiting for batch completion...")
    while batch_response.status not in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED):
        batch_response = vlm.retrieve_batch(batch_response.batch_id)
        time.sleep(15)

    print(f"Batch finished with status: {batch_response.status}")

    if batch_response.status == BatchStatus.COMPLETED:
        # Download result file
        result_file = vlm.download_result(batch_response.batch_id)

        # Read file
        print("Results:")
        for result in vlm.iter_batch_results(result_file) 
            print(result)

    print("Done.")
