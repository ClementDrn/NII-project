"""Exercise real tasks, parsing and dispatch with scripted model responses."""
import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from digit.core.base import CONFIG
from digit.core.settings import GeneratorConfig, GeneratorStepConfig, ModelInstanceConfig
from digit.generator.instruction import Instruction, InstructionResult, InjectionWindow
from digit.generator.model import ModelClientFactory
from digit.generator.model_client import ModelClient, ModelType
from digit.generator.multi_step_vlm_instruction_generator import MultiStepVLMInstructionGenerator
from digit.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
from digit.generator.request import ImageBlock, TextBlock
from tests.fakes import FakeModelClient


GENERATORS = [SingleStepVLMInstructionGenerator, MultiStepVLMInstructionGenerator]
RESPONSE = json.dumps([{
    'rank': 1, 'injection_window': {'start_timestamp': 1.0, 'end_timestamp': 2.0},
    'text': 'Turn left at the intersection', 'target_module_logic': 'path planning',
    'reasoning': 'A vehicle is approaching',
}])


def text_of(request):
    return '\n'.join(block.text for block in request.user_content if isinstance(block, TextBlock))


@pytest.fixture
def build_generator(monkeypatch):
    def build(generator_type, *, response=RESPONSE, parallel=True, incremental=False):
        multi = generator_type is MultiStepVLMInstructionGenerator
        step_ids = ['descriptor', 'planner'] if multi else ['planner']
        steps = [GeneratorStepConfig(id=step_id, model_name='r4b') for step_id in step_ids]
        CONFIG.generator = GeneratorConfig(
            id='test', design='multi_step_vlm' if multi else 'single_step_vlm',
            steps=steps, is_incremental=incremental, max_previous_instructions=1,
        )
        instances = [ModelInstanceConfig(
            id=step_id, model_name='r4b', type='vlm', step_ids=[step_id],
        ) for step_id in step_ids]
        CONFIG.active_model_instances = instances
        planner = FakeModelClient([response], parallel=parallel)
        clients = {'planner': planner}
        if multi:
            clients['descriptor'] = FakeModelClient([
                '<think>reasoning</think>\n\nA red car is ahead.',
                'A pedestrian is on the right.',
            ], parallel=parallel)

        def create(instance, api_parameters):
            assert instance in instances
            assert api_parameters['model'] == 'yannqi/r-4b'
            return clients[instance.id]

        monkeypatch.setattr(ModelClientFactory, 'create', create)
        return generator_type(is_incremental=incremental), clients
    return build


@asynccontextmanager
async def running(generator):
    await generator.init()
    try:
        yield
    finally:
        await generator.free()


@pytest.mark.asyncio
@pytest.mark.parametrize('generator_type', GENERATORS)
@pytest.mark.parametrize('with_reasoning', [False, True])
async def test_generates_instructions_from_all_frames(build_generator, simulation_data,
                                                     generator_type, with_reasoning):
    response = '<think>details</think>\n\n' + RESPONSE if with_reasoning else RESPONSE
    generator, clients = build_generator(generator_type, response=response)
    async with running(generator):
        result = await asyncio.wait_for(generator.generate_instructions(simulation_data), timeout=2)
        assert [instruction.text for instruction in result.instructions] == ['Turn left at the intersection']
        assert result.instructions[0].injection_window == InjectionWindow(start_timestamp=1, end_timestamp=2)
        assert len(clients['planner'].requests) == 1
        if 'descriptor' in clients:
            descriptor_requests = clients['descriptor'].requests
            assert len(descriptor_requests) == 2
            images = [block.image_path for request in descriptor_requests
                      for block in request.user_content if isinstance(block, ImageBlock)]
            planner_text = text_of(clients['planner'].requests[0])
            assert 'A red car is ahead.' in planner_text
            assert 'A pedestrian is on the right.' in planner_text
            assert planner_text.index('A red car') < planner_text.index('A pedestrian')
            assert '<think>' not in planner_text
            assert not any(isinstance(block, ImageBlock) for block in clients['planner'].requests[0].user_content)
        else:
            images = [block.image_path for block in clients['planner'].requests[0].user_content
                      if isinstance(block, ImageBlock)]
        assert images == [simulation_data.image_dir / frame.image_name for frame in simulation_data.frames]
    assert all(client.get_state() is ModelClient.State.DISABLED for client in clients.values())
    assert all(client.stop_calls == 1 for client in clients.values())


@pytest.mark.asyncio
@pytest.mark.parametrize('generator_type', GENERATORS)
@pytest.mark.parametrize('response', ['REDACTED', 'not valid JSON', '[]'])
async def test_invalid_or_redacted_output_returns_empty_set(build_generator, simulation_data,
                                                           generator_type, response):
    generator, _ = build_generator(generator_type, response=response)
    async with running(generator):
        result = await asyncio.wait_for(generator.generate_instructions(simulation_data), timeout=2)
        assert result.instructions == []


@pytest.mark.asyncio
@pytest.mark.parametrize('generator_type', GENERATORS)
async def test_empty_frames_are_rejected_before_model_request(build_generator, simulation_data, generator_type):
    generator, clients = build_generator(generator_type)
    simulation_data.frames = []
    simulation_data.frame_count = 0
    async with running(generator):
        with pytest.raises(ValueError, match='At least one frame'):
            await generator.generate_instructions(simulation_data)
    assert all(client.requests == [] for client in clients.values())


@pytest.mark.asyncio
@pytest.mark.parametrize('generator_type', GENERATORS)
async def test_model_failure_reaches_caller(build_generator, simulation_data, generator_type):
    generator, _ = build_generator(generator_type, response=RuntimeError('provider failed'))
    async with running(generator):
        with pytest.raises(RuntimeError, match='provider failed'):
            await asyncio.wait_for(generator.generate_instructions(simulation_data), timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize('generator_type', GENERATORS)
async def test_full_speed_mode_activates_and_releases_exclusive_clients(build_generator, simulation_data,
                                                                      generator_type):
    generator, clients = build_generator(generator_type, parallel=False)
    async with running(generator):
        assert all(client.start_calls == 0 for client in clients.values())
        await generator.enable_full_speed_mode()
        result = await asyncio.wait_for(generator.generate_instructions(simulation_data), timeout=2)
        assert len(result.instructions) == 1
        await generator.disable_full_speed_mode()
        assert all(client.get_state() is ModelClient.State.DISABLED for client in clients.values())
    assert all(client.start_calls == 1 and client.stop_calls == 1 for client in clients.values())


@pytest.mark.asyncio
async def test_incremental_generation_keeps_only_recent_history(build_generator, simulation_data):
    generator, clients = build_generator(SingleStepVLMInstructionGenerator, incremental=True)
    history = [InstructionResult(instruction=Instruction(
        rank=1, text=text, injection_window=InjectionWindow(start_timestamp=1, end_timestamp=2),
        reasoning='test', target_module_logic='planning',
    ), was_successful=False) for text in ['old instruction', 'recent instruction']]
    async with running(generator):
        await asyncio.wait_for(generator.generate_instructions(
            simulation_data, previously_tried_instructions=history), timeout=2)
    prompt_text = text_of(clients['planner'].requests[0])
    assert 'recent instruction' in prompt_text
    assert 'old instruction' not in prompt_text


@pytest.mark.parametrize(('generator_type', 'step_count'), [
    (SingleStepVLMInstructionGenerator, 0), (SingleStepVLMInstructionGenerator, 2),
    (MultiStepVLMInstructionGenerator, 1), (MultiStepVLMInstructionGenerator, 3),
])
def test_wrong_step_count_is_rejected(generator_type, step_count):
    CONFIG.generator = GeneratorConfig(id='test', steps=[
        GeneratorStepConfig(id=str(index), model_name='r4b') for index in range(step_count)
    ])
    with pytest.raises(ValueError, match='requires exactly'):
        generator_type()
