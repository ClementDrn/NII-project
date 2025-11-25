from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.vlm import VLM, parse_vlm_response
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData


class SingleStepVLMInstructionGenerator(InstructionGenerator):
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

    def generate_instruction_from_images(self, data: SimulationData) -> str:
        """
        Generate instruction from images and metadata using a VLM.
        """
        # WARN: Currently only a single image and metadata is supported.
        if data.frame_count != 1:
            LOG.error("Currently only single image and metadata is supported!")
            raise NotImplementedError("Currently only single image and metadata is supported.")

        # Prepare metadata text
        metadata_text = str(data.frames[0].metadata)

        # Prepare image path
        image_path = data.image_dir / data.frames[0].image_name

        # Send prompt to VLM
        chat_response = self.vlm.prompt(self.system_task_message, metadata_text, [image_path])

        # Parse the instruction from the response
        instruction = parse_vlm_response(chat_response)

        # The VLM response is the generated text
        return instruction
