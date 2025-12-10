from untitled.core.base import LOG, CONFIG
from untitled.generator.vlm import VLM

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
    # Set up VLM
    vlm_config = CONFIG.generator.models[0]
    vlm = VLM(host=vlm_config.host, api_key=vlm_config.api_key, api_parameters=vlm_config.get_api_parameters())

    # Create a list of coroutines for sending requests
    tasks = [send_request(vlm, image_path) for image_path in images]

    # Run tasks concurrently with 'asyncio.gather()'
    start_time = time.time()
    responses = await asyncio.gather(*tasks)
    end_time = time.time()

    # Print the raw response
    LOG.info("Raw VLM Response:")
    for i, response in enumerate(responses):
        LOG.info(f"[Response {i+1}] - {response}")

    LOG.info(f"All requests completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
