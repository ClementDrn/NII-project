from untitled.generator.instruction_generator import InstructionGenerator
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG

import base64
from PIL import Image
from openai import OpenAI
import io
import time


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image to a base64 string.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        return data_url


class VLM(InstructionGenerator):
    def __init__(self):
        super().__init__()

        # Set OpenAI's API key and API base to use vLLM's API server.
        # NOTE: VLM interface is hardcoded as OpenAI-compatible for now.
        openai_api_base = CONFIG.vlm.host + "/v1"
        self.client = OpenAI(
            api_key=CONFIG.vlm.api_key,
            base_url=openai_api_base,
        )

        # Fetch system task message from its asset file
        SYSTEM_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "system_task.txt"
        with open(SYSTEM_TASK_PATH, "r") as f:
            self.system_task_message = f.read()
        LOG.debug(f"Loaded system task message for VLM. See below:\n{self.system_task_message}")

    def generate_instruction_from_images(self, data) -> str:
        """
        Generate instruction from images and metadata using a VLM.
        """
        # Encode image to base64
        data_url = encode_image_to_base64(PROJECT_ROOT / "assets" / "images" / "example_frame.jpg")

        # Create prompt for VLM
        messages = [
            {"role": "system", "content": self.system_task_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        },
                    },
                    {"type": "text", "text": "{'speed': 0}"},
                ],
            },
        ]

        # Send prompt to VLM and get response
        LOG.debug(f"Sending messages to VLM: {messages}")
        
        # Thinking mode
        # NOTE: R-4B thinking modes: "short" (non-thinking), "auto" (auto-thinking), "long" (thinking)
        thinking_mode = "short" if CONFIG.vlm.enable_thinking == "false" else (
            "long" if CONFIG.vlm.enable_thinking == "true" else "auto"
        )

        start_time = time.time()
        # NOTE: For now, we hardcode parameters for openai client and r4b. Later, parameters will need to depend on config the, as with the values. 
        chat_response = self.client.chat.completions.create(
            model=CONFIG.vlm.model,
            messages=messages,
            max_tokens=CONFIG.vlm.max_tokens,
            temperature=CONFIG.vlm.temperature,
            extra_body={
                "chat_template_kwargs": {
                    "thinking_mode": thinking_mode      # NOTE: This parameter is specific to R-4B
                },
            },
        )
        elapsed_time = time.time() - start_time
        LOG.info(f"VLM request completed in {elapsed_time:.2f} seconds")
        LOG.debug(f"VLM response: {chat_response}")

        # Parse the instruction from the response
        # The instruction is after "</think>\n\n"
        output_text = chat_response.choices[0].message.content.strip()
        instruction = output_text.split("</think>\n\n", 1)[-1].strip()

        # The VLM response is the generated text
        return instruction
