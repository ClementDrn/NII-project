from untitled.core.base import LOG
from untitled.simulator.simulator import Simulator
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.vlm_instruction_generator import VLMInstructionGenerator

class Pipeline:
    """
    Pipeline to run simulation with VLM-generated instructions.
    """

    def __init__(self):
        # Init simulator
        self.simulator = None
        self.generator = None

    def init(self):
        """
        Initialize all pipeline components.
        """
        # Initialize simulator
        self.simulator = Simulator()

        # Initialize VLM
        self.generator = VLMInstructionGenerator()

    def stop(self):
        """
        Stop and free all pipeline components.
        """
        # Free simulator
        if self.simulator:
            self.simulator.free()
            self.simulator = None

        # Free VLM
        if self.generator:
            self.generator.free()
            self.generator = None

    def start(self):
        """
        Start the pipeline execution.
        """
        timestep = 2.0                  # 2 sec
        frame_index = timestep * 20     # 20 fps

        # Run scenario once
        self.simulator.run_scenario()  # NOTE: Scenario number is, for now, hardcoded in the method.

        # Fetch results
        image_path = self.simulator.get_image_log(timestep)
        metadata = self.simulator.get_metadata_log(timestep)
        LOG.info(f"Fetched image at: {image_path}")
        LOG.info(f"Fetched metadata: {metadata}")

        if image_path is None:
            LOG.error("Failed to get image from simulation. Cannot continue.")
            return
        
        simulation_data = SimulationData(
            frame_count=1,
            image_dir=image_path.parent,
            images=[image_path.name],
            metadata=[metadata]
        )
        
        # Give data to VLM to generate instruction
        instruction = self.generator.generate_instruction_from_images(simulation_data)
        if not instruction:
            LOG.error("Failed to generate instruction from VLM. Cannot continue.")
            return
        LOG.info(f"Generated instruction: {instruction}")

        # Run scenario again with generated instruction
        self.simulator.run_scenario(instruction=instruction, injection_time=timestep)

        # Fetch new results
        new_image_path = self.simulator.get_image_log(timestep)
        new_metadata = self.simulator.get_metadata_log(timestep)


