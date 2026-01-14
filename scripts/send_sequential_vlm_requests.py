from untitled.core.base import LOG, CONFIG
from untitled.core.settings import get_model_instances_config_by_step_id
from untitled.generator.model import create_model_client
from untitled.generator.instruction_generator import get_api_parameters_from_config

import asyncio
from pathlib import Path
import time


# --- SET ARGUMENTS BELOW FOR TESTING PURPOSES ---
# Input images
images = [
    Path("cache/sce_005/run_000/images/000005.png"),
    Path("cache/sce_005/run_000/images/000010.png"),
    Path("cache/sce_005/run_000/images/000015.png"),
    Path("cache/sce_005/run_000/images/000020.png"),
    Path("cache/sce_005/run_000/images/000025.png"),
]

# Text prompts
system_prompt = "You are a helpful assistant that analyzes images."
user_prompt = "Describe the contents of this image."
# ------------------------------------------------


async def send_request(vlm, image_path):
    # Send request to VLM and log time taken
    start_time = time.time()
    response = await vlm.prompt(system_text=system_prompt, user_text=user_prompt, user_image_paths=[image_path])
    end_time = time.time()
    LOG.info(f"Request for {image_path} completed in {end_time - start_time:.2f} seconds.")
    return response

async def main():
    # Set up VLM client
    model_config = get_model_instances_config_by_step_id(CONFIG, "description")[0]
    api_params = get_api_parameters_from_config(CONFIG.generator.steps[0])
    vlm = create_model_client(model_config, api_params)
    await vlm.start()

    try:
        # Open connection to VLM
        await vlm.open_connection()

        # Send requests to VLM
        responses = []
        
        start_time = time.time()
        for image_path in images:
            # Send request to VLM and log time taken
            response = await vlm.prompt(system_text=system_prompt, user_text=user_prompt, user_image_paths=[image_path])
            responses.append(response)
        end_time = time.time()

        # Print the raw response
        LOG.info("Raw VLM Response:")
        for i, response in enumerate(responses):
            LOG.info(f"[Response {i+1}] - {response}")

        LOG.info(f"All requests completed in {end_time - start_time:.2f} seconds.")

        # Close connection
        await vlm.close_connection()
    except Exception as e:
        LOG.error(e, exc_info=True)

    # Finally, stop the VLM client
    await vlm.stop()


if __name__ == "__main__":
    asyncio.run(main())
