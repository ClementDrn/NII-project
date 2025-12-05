from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.vlm import VLM, parse_vlm_response
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from typing import Optional


class SingleStepVLMInstructionGenerator(InstructionGenerator):
    """
    Vision-Language Model (VLM) interface for generating instructions from images and metadata.
    The actual VLM process should be running as a separate service, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        # Check that there is exactly one model configured
        if len(CONFIG.generator.models) != 1:
            LOG.error("Single-step VLM instruction generator requires exactly one model configured in 'generator.models' field!")
            raise ValueError("Single-step VLM instruction generator requires exactly one model configured.")
        
        host = CONFIG.generator.models[0].host
        api_key = CONFIG.generator.models[0].api_key
        self.vlm = VLM(host, api_key, CONFIG.generator.models[0].get_api_parameters())

        # Fetch system task message from its asset file
        SYSTEM_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "single_frame_adversarial_generator_task.txt"
        with open(SYSTEM_TASK_PATH, "r") as f:
            self.system_task_message = f.read()
        LOG.debug(f"Loaded system task message for VLM. See below:\n{self.system_task_message}")

    def free(self):
        pass

    def generate_instructions_from_images(self, data: SimulationData) -> Optional[InstructionSet]:
        """
        Generate instruction from images and metadata using a VLM.
        """
        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # NOTE: Currently only a single image and metadata is supported. It is hardcoded to use the frame at timestamp 2.0s.
        fixed_timestamp = 2.0
        fixed_frame_index = int(fixed_timestamp * 20)  # Assuming 20 FPS
        
        # If the fixed frame index is out of range, use the first frame
        if fixed_frame_index >= data.frame_count:
            LOG.warning(f"Fixed frame index {fixed_frame_index} is out of range for {data.frame_count} frames. Using first frame instead.")
            fixed_frame_index = 0
            fixed_timestamp = 0.0

        # Prepare metadata text and image path
        metadata_text = str(data.frames[fixed_frame_index].metadata)
        image_path = data.image_dir / data.frames[fixed_frame_index].image_name

        # Send prompt to VLM
        response_text = self.vlm.prompt(self.system_task_message, metadata_text, [image_path])

        # Remove the thinking part from response
        instruction_raw = parse_vlm_response(response_text)
        if instruction_raw is None:
            return None
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)

        # Hardset the timestep in each instruction
        for instruction in instruction_set.instructions:
            instruction.injection_window.start_timestamp = fixed_timestamp
            instruction.injection_window.end_timestamp = fixed_timestamp + 0.1  # Small window after fixed timestamp

        return instruction_set
