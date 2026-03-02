from untitled.core.base import LOG, CONFIG
from untitled.core.settings import HostConfig, ModelInstanceConfig, get_model_instance_config_by_id, get_host_config_by_id
from untitled.generator.model_config import ModelConfig, get_model_config
from untitled.generator.model_client import ModelType, parse_model_response
from untitled.generator.remote_model import RemoteModel

import asyncio


# --- SET ARGUMENTS BELOW FOR TESTING PURPOSES ---
MODEL_INSTANCE_ID = "gemini-3.1-pro-preview_0"  # from config.yaml model_instances section
SYSTEM_PROMPT = "You are an expert in problem solving and answer questions accurately."
USER_PROMPT = "If it takes 1 hour to dry one towel outside in the sun, how long does it take to dry 5 towels?"
# ------------------------------------------------


async def main():
    # Get model instance configuration
    instance_config: ModelInstanceConfig = get_model_instance_config_by_id(CONFIG, MODEL_INSTANCE_ID)
    model_config: ModelConfig = get_model_config(instance_config.model_name)

    # Get host credentials
    host_config: HostConfig = get_host_config_by_id(CONFIG, instance_config.remote_config.host_id)
    host = host_config.host
    api_key = host_config.api_key

    # Prepare API parameters
    no_thinking_params = model_config.thinking_deactivation_params if model_config.thinking_deactivation_params is not None else {}
    api_params_raw = {
        "model": model_config.model_name,
        model_config.max_output_tokens_param: 200 if model_config.max_output_tokens_param else None,    # 200 max tokens for testing
        model_config.temperature_param: 1.0 if model_config.temperature_param else None,                # temperature 1
        **no_thinking_params                                                                            # no thinking
    }
    api_params = {key: value for key, value in api_params_raw.items() if key and value is not None}

    # Create LLM client
    llm = RemoteModel(
        model_type=ModelType.LLM,
        priority=1,
        batch_size=1,
        host=host,
        api_key=api_key,
        api_parameters=api_params
    )
    await llm.start()
    
    try:
        # Open connection to LLM
        await llm.open_connection()

        # Send request to LLM
        response = await llm.prompt(system_text=SYSTEM_PROMPT, user_text=USER_PROMPT, user_image_paths=None)

        # Print the raw response
        LOG.info("Raw LLM Response:")
        LOG.info(response)
        LOG.info("Parsed LLM Response:")
        LOG.info(parse_model_response(response))

        # Stop the LLM client
        await llm.close_connection()
    except Exception as e:
        LOG.error(e, exc_info=True)

    # Finally, stop the LLM client
    await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())
