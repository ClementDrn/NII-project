from untitled.core.base import LOG, CONFIG
from untitled.core.settings import get_model_instances_config_by_step_id
from untitled.generator.model import create_model_client
from untitled.generator.instruction_generator import get_api_parameters_from_config

from pathlib import Path
import asyncio


# --- SET ARGUMENTS BELOW FOR TESTING PURPOSES ---
# Input images
images = [
    Path("cache/sce_005/run_000/images/000005.png"),
    Path("cache/sce_005/run_000/images/000300.png"),
    Path("cache/sce_005/run_000/images/000965.png"),
]

# Text prompts
system_prompt = "You are a helpful assistant that analyzes images."
user_prompt = "Describe the contents of the first, second, and third images."
# ------------------------------------------------


async def main():
    # Get first VLM model instance for description step
    model_config = get_model_instances_config_by_step_id(CONFIG, "description")[0]
    api_params = get_api_parameters_from_config(CONFIG.generator.steps[0])

    # Create VLM client
    vlm = create_model_client(model_config, api_params)
    await vlm.start()
    
    try:
        # Open connection to VLM
        await vlm.open_connection()

        # Send request to VLM
        response = await vlm.prompt(system_text=system_prompt, user_text=user_prompt, user_image_paths=images)

        # Print the raw response
        LOG.info("Raw VLM Response:")
        LOG.info(response)

        # Close connection
        await vlm.close_connection()
    except Exception as e:
        LOG.error(e, exc_info=True)

    # Finally, stop the VLM client
    await vlm.stop()


if __name__ == "__main__":
    asyncio.run(main())
