from untitled.core.base import LOG, CONFIG
from untitled.generator.vlm import VLM

from pathlib import Path


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

vlm_config = CONFIG.generator.models[0]

# Send request to VLM
vlm = VLM(host=vlm_config.host, api_key=vlm_config.api_key, api_parameters=vlm_config.get_api_parameters())
response = vlm.prompt(system_text=system_prompt, user_text=user_prompt, user_image_paths=images)

# Print the raw response
LOG.info("Raw VLM Response:")
LOG.info(response)
