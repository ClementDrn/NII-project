from untitled.generator.instruction_generator import InstructionGenerator
from untitled.generator.instruction_pool import InstructionPool, InstructionPoolCategory
from untitled.generator.instruction import InjectionWindow, Instruction, InstructionCategory, InstructionResult, InstructionSet
from untitled.core.base import LOG, VENDOR_LAMPILOT_DIR

import json
import random
import re
from pathlib import Path
from typing import List, Optional, override

from untitled.generator.rules import get_rule_for_category
from untitled.simulator.simulation_data import SimulationData


class RuleBasedInstructionGenerator(InstructionGenerator):
    """
    A simple instruction generator that uses a rule-based approach to generate instructions based on the current state of the environment and the agent.
    """
    def __init__(self, dataset: str = ""):
        super().__init__()
        self._dataset = dataset
        self._instruction_pool = self.load_instruction_pool(dataset)

    def load_instruction_pool(self, dataset: str) -> InstructionPool:
        """
        Load and instantiate the instruction pool from a named dataset.
        """
        if not dataset:
            raise ValueError("Rule-based dataset name must be provided.")

        # The only supported dataset is "lampilot-bench" for now
        dataset_key = dataset.strip().lower()
        if dataset_key != "lampilot-bench":
            raise ValueError(f"Unsupported rule-based dataset: '{dataset}'")

        data_dir = VENDOR_LAMPILOT_DIR
        if not data_dir.exists():
            raise FileNotFoundError(f"Rule-based dataset directory not found: {data_dir}")

        pool = InstructionPool()
        loaded_files = 0
        skipped_files = 0

        # Load dataset (json files)
        for json_path in sorted(data_dir.glob("*.json")):
            parsed = self._parse_category_from_file_stem(json_path.stem)
            if parsed is None:
                skipped_files += 1
                continue

            category, variable = parsed
            commands = self._read_commands_from_file(json_path)
            if not commands:
                skipped_files += 1
                continue

            pool.add_instructions(
                InstructionPoolCategory(category=category, variable=variable),
                commands,
            )
            loaded_files += 1

        if not pool.get_categories():
            raise RuntimeError(
                f"No instruction categories could be loaded from dataset '{dataset}' in {data_dir}"
            )

        LOG.info(
            f"Loaded rule-based instruction pool from {dataset}: {loaded_files} files, "
            f"{len(pool.get_categories())} categories ({skipped_files} skipped)"
        )
        return pool

    def _read_commands_from_file(self, json_path: Path) -> List[str]:
        """
        Read the list of instruction commands from a given JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # The commands must be in {commands: [list of strings]} format
        commands = data.get("commands", [])
        if not isinstance(commands, list):
            LOG.warning(f"Invalid format for commands in {json_path}")
            return []
        return [cmd.strip() for cmd in commands if isinstance(cmd, str) and cmd.strip()]

    def _parse_category_from_file_stem(self, stem: str) -> Optional[tuple[InstructionCategory, int]]:
        """
        Parse the instruction category and variable (if applicable) from the file stem.
        """
        # Without variables
        direct_mappings: dict[str, tuple[InstructionCategory, int]] = {
            "left_lc": (InstructionCategory.LANE_CHANGE_LEFT, 0),
            "right_lc": (InstructionCategory.LANE_CHANGE_RIGHT, 0),
            "left_overtake": (InstructionCategory.OVERTAKE_LEFT, 0),
            "right_overtake": (InstructionCategory.OVERTAKE_RIGHT, 0),
            "turn_left": (InstructionCategory.TURN_LEFT, 0),
            "turn_right": (InstructionCategory.TURN_RIGHT, 0),
            "go_straight": (InstructionCategory.GO_STRAIGHT, 0),
            "pull_over": (InstructionCategory.PULL_OVER, 0),
        }
        if stem in direct_mappings:
            return direct_mappings[stem]

        # With variables (e.g., inc_abs_speed30)
        pattern_mappings: list[tuple[str, InstructionCategory]] = [
            (r"^inc_abs_speed(\d+)$", InstructionCategory.INCREASE_ABSOLUTE_SPEED),
            (r"^dec_abs_speed(\d+)$", InstructionCategory.DECREASE_ABSOLUTE_SPEED),
            (r"^inc_rel_speed(\d+)$", InstructionCategory.INCREASE_RELATIVE_SPEED),
            (r"^dec_rel_speed(\d+)$", InstructionCategory.DECREASE_RELATIVE_SPEED),
            (r"^inc_abs_dis(\d+)$", InstructionCategory.INCREASE_ABSOLUTE_DISTANCE),
            (r"^dec_abs_dis(\d+)$", InstructionCategory.DECREASE_ABSOLUTE_DISTANCE),
            (r"^inc_rel_dis(\d+)$", InstructionCategory.INCREASE_RELATIVE_DISTANCE),
            (r"^dec_rel_dis(\d+)$", InstructionCategory.DECREASE_RELATIVE_DISTANCE),
        ]

        for pattern, category in pattern_mappings:
            match = re.match(pattern, stem)
            if match:
                return (category, int(match.group(1)))

        LOG.debug(f"Skipping unsupported instruction dataset file: {stem}")
        return None

    def _determine_instruction_category(self, instruction_category: InstructionCategory) -> Optional[InstructionPoolCategory]:
        """
        Determine the instruction text based on the category, using the instruction pool.
        """
        categories = self._instruction_pool.get_categories()
        if not categories:
            LOG.error("Instruction pool is empty. No categories available.")
            return None

        category: InstructionPoolCategory
        # Select random category if none specified
        if instruction_category == InstructionCategory.NONE:
            category = random.choice(categories)
        # Otherwise, find matching categories and select one
        else:
            base_categories = {
                InstructionCategory.SPEED_CHANGE,
                InstructionCategory.DISTANCE_CHANGE,
                InstructionCategory.OVERTAKE,
                InstructionCategory.TURN,
                InstructionCategory.LANE_CHANGE,
            }
            if instruction_category in base_categories:
                matching_categories = [c for c in categories if c.category.is_part_of(instruction_category)]
            else:
                matching_categories = [c for c in categories if c.category == instruction_category]
            # If no matching categories found, return empty instruction set
            if not matching_categories:
                LOG.error(f"No matching instruction categories found for specified category: {instruction_category}")
                return None
            # If multiple matching categories found, select one at random
            category = random.choice(matching_categories)
            
        return category

    def _determine_instruction_text(self, category: InstructionPoolCategory) -> str:
        """
        Determine the instruction text based on the category, using the instruction pool.
        """
        # Get instructions for the selected category
        instruction_texts = self._instruction_pool.get_instructions_by_category(category)
        if not instruction_texts:
            LOG.error(f"No instructions found in pool for category: {category}")
            return ""
        
        # Choose randomly from the available instructions in the category
        selected_instruction_text = random.choice(instruction_texts)
        return selected_instruction_text

    @override
    async def generate_instructions(
        self, 
        data: SimulationData,
        failure_text: Optional[str] = None,
        was_route_completed: Optional[bool] = None,
        previously_tried_instructions: Optional[List[InstructionResult]] = None,
        instruction_category: InstructionCategory = InstructionCategory.NONE,
        timestamp: Optional[float] = None,
    ) -> InstructionSet:
        """
        Generate instructions from simulation data (async to support VLM calls).
        
        Args:
            data: Simulation data containing frames and metadata
            failure_text: Description of failure that occurred (for corrective tasks)
            was_route_completed: Whether route was completed (for corrective tasks)
            previously_tried_instructions: List of previous instruction results
            instruction_category: Optional category to guide instruction generation
            timestamp: Optional timestamp to force a specific injection time
        Returns:
            Generated instruction set
        """
        # 1. Determine the instruction text based on the provided category
        category: Optional[InstructionPoolCategory] = self._determine_instruction_category(instruction_category)
        if category is None:
            LOG.error("Failed to determine instruction category. No instructions will be generated.")
            return InstructionSet(instructions=[])
        selected_instruction_text = self._determine_instruction_text(category)
        if not selected_instruction_text:
            LOG.error("Failed to determine instruction text. No instructions will be generated.")
            return InstructionSet(instructions=[])

        # 2. Get the rule for the selected category
        rule = get_rule_for_category(category.category)
        if rule is None:
            LOG.error(f"No rule found for instruction category: {category}. No instructions will be generated.")
            return InstructionSet(instructions=[])
        
        # 3. Determine the injection window, based on the rule
        injection_window: InjectionWindow
        # If timestamp is forced, check that the rule is still valid at that timestamp
        if timestamp is not None:
            if rule.is_valid_for_this_frame(data.frames[0], category.variable):
                injection_window = InjectionWindow(start_timestamp=timestamp, end_timestamp=timestamp + rule.injection_duration)
            else:
                LOG.error(f"Forced timestamp {timestamp} is not valid for the selected instruction category {category}. No instructions will be generated.")
                return InstructionSet(instructions=[])
        # If no timestamp is forced, determine injection window based on the rule
        else:
            # Find all frames that validate the rule's conditions
            possible_frames = [frame for frame in data.frames if rule.is_valid_for_this_frame(frame, category.variable)]
            if not possible_frames:
                LOG.error(f"No valid frames found for the selected instruction category {category}. No instructions will be generated.")
                return InstructionSet(instructions=[])
            
            # Choose a random valid frame
            chosen_frame = random.choice(possible_frames)
            injection_window = InjectionWindow(
                start_timestamp=chosen_frame.timestamp, 
                end_timestamp=chosen_frame.timestamp + rule.injection_duration
            )
        
        # Return only one instruction
        return InstructionSet(instructions=[
            Instruction(
                rank=1,
                text=selected_instruction_text,
                injection_window=injection_window,
                target_module_logic=rule.target_module_logic,
                reasoning=rule.reasoning,
                category=category.category,
            )
        ])

    @override
    async def enable_full_speed_mode(self):
        """
        Enable full speed operation (async to support VLM clients).
        """
        # Unrelevant
        pass

    @override
    async def disable_full_speed_mode(self):
        """
        Disable full speed operation (async to support VLM clients).
        """
        # Unrelevant
        pass

    @override
    def get_supported_instruction_categories(self) -> List[InstructionCategory]:
        """
        Get the list of instruction categories supported by this generator.
        
        Returns:
            List of supported instruction categories
        """
        # Extract all unique instruction categories from the instruction pool
        plain_categories = []
        for category in self._instruction_pool.get_categories():
            if category.category not in plain_categories:
                plain_categories.append(category.category)
        return plain_categories