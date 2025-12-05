from untitled.generator.instruction_generator import InstructionGenerator
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData

import base64
from PIL import Image
from openai import OpenAI
import io
import time
from pathlib import Path


def encode_image_to_base64(image_path: Path) -> str:
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

def parse_vlm_response(chat_response: str) -> str:
    """
    Parse the VLM response to extract the relevant content after the reasoning text.
    """
    # We do not want the thinking part, before "</think>\n\n"
    # WARN: Specific to r4b model response format
    split_response = chat_response.split("</think>\n\n", 1)
    after_reasoning = split_response[-1].strip()
    return after_reasoning


class VLM:
    """
    Interface to a Vision-Language Model (VLM) service via OpenAI-compatible API.
    """

    def __init__(self, host, api_key, api_parameters = {}):
        super().__init__()

        # Get API parameters from config
        # Parameters my include model, temperature, max_tokens, etc.
        self.api_parameters = api_parameters
        LOG.debug(f"VLM API parameters: {self.api_parameters}")

        # Set OpenAI's API key and API base to use vLLM's API server.
        # NOTE: VLM interface is hardcoded as OpenAI-compatible for now.
        openai_api_base = host + "/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=openai_api_base,
        )

    def prompt(self, system_text: str, user_text: str, user_image_paths: list[Path]) -> str:
        """
        Send a prompt to the VLM and get the response.
        """
        # Append system text to messages
        messages = [
            {"role": "system", "content": system_text},
        ]

        # Append user images and text to messages
        user_content = []   # List to hold user messages (text and images)
        if user_image_paths is not None:
            for image_path in user_image_paths:
                data_url = encode_image_to_base64(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    },
                })
        if user_text is not None:
            user_content.append({
                "type": "text",
                "text": user_text,
            })
        messages.append({
            "role": "user",
            "content": user_content,
        })
        LOG.debug(f"Messages that will be sent to VLM: {messages}")
        
        # Send request to VLM and get response
        # Processing time is measured for performance monitoring
        start_time = time.time()
        chat_response = self.client.chat.completions.create(
            messages=messages,
            **self.api_parameters
        )
        elapsed_time = time.time() - start_time
        LOG.info(f"VLM request completed in {elapsed_time:.2f} seconds")
        LOG.debug(f"VLM response: {chat_response}")

        # Extract and return the response text
        response_text = chat_response.choices[0].message.content
        return response_text
