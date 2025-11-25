from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.vlm import VLM, parse_vlm_response
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData


class MultiStepVLMInstructionGenerator(InstructionGenerator):
    """
    Vision-Language Model (VLM) interface for generating instructions from images and metadata.
    The actual VLM process should be running as a separate service, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        # Check that there are exactly two models configured
        if len(CONFIG.generator.models) != 2:
            LOG.error("Multi-step VLM instruction generator requires exactly two models configured in 'generator.models' field!")
            raise ValueError("Multi-step VLM instruction generator requires exactly two models configured.")
        
        # Initialize two VLM instances for the two steps, with their respective configurations
        # NOTE: Only 1 client would be enough, but in this case API parameters are provided at each request (like for the task prompt).
        # NOTE:  
        self.descriptor = VLM(CONFIG.generator.models[0].get_api_parameters())
        self.planner = VLM(CONFIG.generator.models[1].get_api_parameters())

        # Fetch scene descriptor task message from its asset file
        SCENE_DESCRIPTOR_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "scene_descriptor_task.txt"
        with open(SCENE_DESCRIPTOR_TASK_PATH, "r") as f:
            self.scene_descriptor_task_message = f.read()
        LOG.debug(f"Loaded scene descriptor task message for VLM. See below:\n{self.scene_descriptor_task_message}")

        # Fetch adversarial planner task message from its asset file
        PLANNER_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "adversarial_planner_task.txt"
        with open(PLANNER_TASK_PATH, "r") as f:
            self.planner_task_message = f.read()
        LOG.debug(f"Loaded adversarial planner task message for VLM. See below:\n{self.planner_task_message}")

    def free(self):
        pass

    def generate_instruction_from_images(self, data: SimulationData) -> str:
        """
        Generate instruction from images and metadata using a VLM.
        There are two steps:
        1. Scene Descriptor: Generate a scene description for each provided frame.
        2. Adversarial Planner: Generate an adversarial instruction based on the concatenated scene descriptions.
        """
        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # Step 1:
        # Generate a scene description for each frame
        scene_descriptions = []
        for i in range(data.frame_count):
            # Prepare prompt inputs
            metadata_text = str(data.frames[i].metadata)
            image_path = data.image_dir / data.frames[i].image_name

            # Send prompt to VLM for scene description
            chat_response = self.descriptor.prompt(self.scene_descriptor_task_message, metadata_text, [image_path])
            scene_description = parse_vlm_response(chat_response)

            # Add description to list
            scene_descriptions.append(scene_description)

        # Concatenate all scene descriptions and add index labels
        concatenated_descriptions = "\n\n".join(
            f"Frame {i+1}:\n{desc}" 
            for i, desc in enumerate(scene_descriptions)
        )

        # Step 2:
        # Generate an adversarial instruction based on the concatenated scene descriptions
        chat_response = self.planner.prompt(self.planner_task_message, concatenated_descriptions, None)
        instruction = parse_vlm_response(chat_response)

        # The VLM response is the generated text
        return instruction
