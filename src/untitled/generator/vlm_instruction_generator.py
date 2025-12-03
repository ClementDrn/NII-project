from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.vlm import VLM
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.instruction import Instruction, parse_instruction_from_text


class VLMInstructionGenerator(InstructionGenerator):
    """
    Vision-Language Model (VLM) interface for generating instructions from images and metadata.
    The actual VLM process should be running as a separate service, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        self.vlm = VLM()

        # Fetch system task message from its asset file
        SYSTEM_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "system_task.txt"
        with open(SYSTEM_TASK_PATH, "r") as f:
            self.system_task_message = f.read()
        LOG.debug(f"Loaded system task message for VLM. See below:\n{self.system_task_message}")

    def free(self):
        pass

    def generate_instruction_from_images(self, data: SimulationData) -> Instruction:
        """
        Generate instruction from images and metadata using a VLM.
        """
        # WARN: Currently only a single image and metadata is supported.
        if data.frame_count != 1:
            LOG.error("Currently only single image and metadata is supported!")
            raise NotImplementedError("Currently only single image and metadata is supported.")

        # Prepare metadata text
        metadata_text = str(data.metadata[0])

        # Prepare image path
        image_path = data.image_dir / data.images[0]

        # Send prompt to VLM
        response_text = self.vlm.prompt(self.system_task_message, metadata_text, [image_path])

        # Parse the instruction from the response
        # We do not want the thinking part, before "</think>\n\n"
        # WARN: This is specific to the R-4B model's response format.
        split_response = response_text.split("</think>\n\n", 1)
        if len(split_response) != 2:
            LOG.warning("VLM response does not contain expected '</think>' delimiter. The token limit may have been reached. Returning an empty instruction.")
            return None
        
        # Parse instruction: "text",category0,category1,...
        instruction_raw = split_response[1].strip()
        instruction = parse_instruction_from_text(instruction_raw)

        return instruction
