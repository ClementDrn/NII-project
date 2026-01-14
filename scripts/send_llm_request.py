from untitled.core.base import LOG, CONFIG
from untitled.core.settings import get_model_instances_config_by_step_id
from untitled.generator.model import create_model_client
from untitled.generator.instruction_generator import get_api_parameters_from_config

from pathlib import Path
import asyncio


# --- SET ARGUMENTS BELOW FOR TESTING PURPOSES ---
# Text prompts
system_prompt = "You are an expert in problem solving and answer questions accurately."
user_prompt = "If it takes 1 hour to dry one towel outside in the sun, how long does it take to dry 5 towels?"
# ------------------------------------------------


async def main():
    model_config = get_model_instances_config_by_step_id(CONFIG, "planning")[0]
    api_params = get_api_parameters_from_config(CONFIG.generator.steps[1])

    # Create LLM client
    llm = create_model_client(model_config, api_params)
    await llm.start()
    
    try:
        # Open connection to LLM
        await llm.open_connection()

        # Send request to LLM
        response = await llm.prompt(system_text=system_prompt, user_text=user_prompt, user_image_paths=None)

        # Print the raw response
        LOG.info("Raw LLM Response:")
        LOG.info(response)

        # Stop the LLM client
        await llm.close_connection()
    except Exception as e:
        LOG.error(e, exc_info=True)

    # Finally, stop the LLM client
    await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())
