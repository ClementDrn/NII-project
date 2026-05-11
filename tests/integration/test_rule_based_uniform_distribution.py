import random
from collections import Counter
from pathlib import Path

import pytest
from unittest.mock import MagicMock

from untitled.core.base import CONFIG
from untitled.core.job_handler import Job
from untitled.core.pipeline import Pipeline
from untitled.core.settings import GeneratorConfig
from untitled.generator.instruction import InstructionCategory
from untitled.generator.rule_based_instruction_generator import RuleBasedInstructionGenerator
from untitled.simulator.simulation_data import SimulationData, SimulationFrameData, SimulationMetadata
import untitled.generator.rule_based_instruction_generator as rule_based_module


class AlwaysValidRule:
    def __init__(self, category: InstructionCategory):
        self.category = category
        self.injection_duration = 1.0
        self.target_module_logic = "test"
        self.reasoning = "test"
        self.is_valid_for_this_frame = lambda frame, var: True


class NeverValidRule(AlwaysValidRule):
    def __init__(self, category: InstructionCategory):
        super().__init__(category)
        self.is_valid_for_this_frame = lambda frame, var: False


def create_simulation_data(image_dir: Path) -> SimulationData:
    metadata = SimulationMetadata(
        step=0,
        control={"throttle": 0.0, "steer": 0.0, "brake": 0.0},
        speed=0.0,
        position=[0.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0],
        forward_vector=[1.0, 0.0, 0.0],
        right_vector=[0.0, 1.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        acceleration=[0.0, 0.0, 0.0],
        scene=[],
    )
    frame = SimulationFrameData(image_name="frame.png", metadata=metadata, timestamp=0.0)
    return SimulationData(image_dir=image_dir, frame_count=1, frames=[frame])


@pytest.mark.asyncio
async def test_rule_based_uniform_distribution_without_carla(tmp_path, monkeypatch):
    random.seed(7)

    generator = RuleBasedInstructionGenerator(dataset="lampilot-bench")

    pipeline = Pipeline()
    pipeline.generator = generator
    pipeline.simulator = MagicMock()
    pipeline.simulator.get_run_dir.side_effect = lambda *args, **kwargs: tmp_path

    generator_config = GeneratorConfig(id="rule_based", design="rule_based", dataset="lampilot-bench")
    monkeypatch.setattr(CONFIG, "generator", generator_config)

    supported_categories = generator.get_supported_instruction_categories()
    pipeline._supported_instruction_categories = supported_categories

    job = Job(
        id=0,
        name="Uniform Distribution",
        simulator="carla_basic",
        scenarios=[0],
        ground_truth_count=1,
        instruction_generation_count=2,
        instruction_count_per_generation=1,
        repeat_for_uniform_instruction_category_distribution=True,
    )

    simulation_data = create_simulation_data(tmp_path)

    async def fake_cluster_frames_for_run(*args, **kwargs):
        return simulation_data

    def fake_get_rule_for_category(category: InstructionCategory):
        return AlwaysValidRule(category)

    monkeypatch.setattr(pipeline, "_cluster_frames_for_run", fake_cluster_frames_for_run)
    monkeypatch.setattr(rule_based_module, "get_rule_for_category", fake_get_rule_for_category)

    counts = Counter()
    category_sequence = []
    repeat_count = len(supported_categories)
    for repeat_index in range(repeat_count):
        instructions = await pipeline._generate_scenario_instructions(
            job=job,
            scenario_id=0,
            previously_tried_instructions=[],
            repeat_index=repeat_index,
        )
        for instruction_set in instructions.values():
            assert instruction_set is not None
            assert instruction_set.instructions
            generated_category = instruction_set.instructions[0].category
            counts[generated_category] += 1
            category_sequence.append(generated_category)

    expected_count = job.instruction_generation_count
    assert set(counts.keys()) == set(supported_categories)
    assert all(count == expected_count for count in counts.values())

    expected_sequence = []
    for repeat_index in range(repeat_count):
        for generation_index in range(job.instruction_generation_count):
            category_index = (repeat_index * job.instruction_generation_count + generation_index) % len(supported_categories)
            expected_sequence.append(supported_categories[category_index])
    assert category_sequence == expected_sequence


@pytest.mark.asyncio
async def test_rule_based_uniform_distribution_skips_invalid_category(tmp_path, monkeypatch):
    random.seed(7)

    generator = RuleBasedInstructionGenerator(dataset="lampilot-bench")

    pipeline = Pipeline()
    pipeline.generator = generator
    pipeline.simulator = MagicMock()
    pipeline.simulator.get_run_dir.side_effect = lambda *args, **kwargs: tmp_path

    generator_config = GeneratorConfig(id="rule_based", design="rule_based", dataset="lampilot-bench")
    monkeypatch.setattr(CONFIG, "generator", generator_config)

    supported_categories = generator.get_supported_instruction_categories()[:2]
    pipeline._supported_instruction_categories = supported_categories

    job = Job(
        id=0,
        name="Uniform Distribution",
        simulator="carla_basic",
        scenarios=[0],
        ground_truth_count=1,
        instruction_generation_count=1,
        instruction_count_per_generation=1,
        repeat_for_uniform_instruction_category_distribution=True,
    )

    simulation_data = create_simulation_data(tmp_path)

    async def fake_cluster_frames_for_run(*args, **kwargs):
        return simulation_data

    invalid_category = supported_categories[0]

    def fake_get_rule_for_category(category: InstructionCategory):
        if category == invalid_category:
            return NeverValidRule(category)
        return AlwaysValidRule(category)

    monkeypatch.setattr(pipeline, "_cluster_frames_for_run", fake_cluster_frames_for_run)
    monkeypatch.setattr(rule_based_module, "get_rule_for_category", fake_get_rule_for_category)

    # Repeat 0: invalid category -> no instruction set
    instructions = await pipeline._generate_scenario_instructions(
        job=job,
        scenario_id=0,
        previously_tried_instructions=[],
        repeat_index=0,
    )
    assert instructions[0] is None

    # Repeat 1: valid category -> generated instruction set
    instructions = await pipeline._generate_scenario_instructions(
        job=job,
        scenario_id=0,
        previously_tried_instructions=[],
        repeat_index=1,
    )
    assert instructions[0] is not None
    assert instructions[0].instructions[0].category == supported_categories[1]
