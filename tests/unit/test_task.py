import asyncio

import pytest
import yaml

from digit.core.base import CONFIG
from digit.core.settings import GeneratorConfig
from digit.generator.request import ImageBlock, TextBlock
from digit.generator.task import Task, TaskData


@pytest.fixture
def task_factory(tmp_path):
    def create(inputs=None, prompt='Test system prompt', **overrides):
        prompt_file = tmp_path / 'prompt.txt'
        prompt_file.write_text(prompt)
        config = {
            'name': 'Test Task', 'description': 'A driving task',
            'system_prompt_path': str(prompt_file), 'inputs': inputs or {},
            'outputs': {'instruction': True},
            **overrides,
        }
        task_file = tmp_path / 'task.yaml'
        task_file.write_text(yaml.safe_dump(config))
        return Task(task_file)
    return create


def test_loads_task_flags_and_system_prompt(task_factory):
    task = task_factory({'images': True, 'metadata': True})
    assert task.name == 'Test Task'
    assert task.description == 'A driving task'
    assert task.get_prompt_system_text() == 'Test system prompt'
    assert task.has_images_input and task.has_metadata_input
    assert not task.has_descriptions_input
    assert not task.has_failure_text_input
    assert not task.has_route_completion_input
    assert task.has_instruction_output
    assert not task.has_descriptions_output
    assert 'Test Task' in repr(task)


def test_system_prompt_uses_active_instruction_count(task_factory):
    CONFIG.generator = GeneratorConfig(id='test', instruction_count_per_generation=4)
    task = task_factory(prompt='Generate ${instruction_count_per_generation}$ instructions.')
    assert task.get_prompt_system_text() == 'Generate 4 instructions.'


@pytest.mark.asyncio
async def test_prompt_preserves_frame_order_and_caches_static_blocks(task_factory, simulation_data):
    task = task_factory({'images': True, 'metadata': True, 'failure_text': True})
    images = [simulation_data.image_dir / frame.image_name for frame in simulation_data.frames]
    future = asyncio.get_running_loop().create_future()
    prompt = task.prepare_prompt(future, images=images, frames=simulation_data.frames,
                                 failure_text='Collision with vehicle')
    assert prompt.future is future
    assert prompt.request.system_text == task.get_prompt_system_text()
    blocks = prompt.request.user_content
    assert len(blocks) == 4
    assert isinstance(blocks[0], TextBlock)
    assert 'Frame 1 (time: 0.0)' in blocks[0].text
    assert 'Frame 2 (time: 1.0)' in blocks[0].text
    assert blocks[1:3] == [ImageBlock(image, to_be_cached=True) for image in images]
    assert all(block.to_be_cached for block in blocks[:3])
    assert blocks[-1] == TextBlock('FAILURE REPORT:\nCollision with vehicle')


@pytest.mark.asyncio
@pytest.mark.parametrize(('completed', 'status'), [(False, 'not completed'), (True, 'completed')])
async def test_corrective_prompt_includes_failure_and_route_status(task_factory, completed, status):
    task = task_factory({'failure_text': True, 'route_completion': True})
    prompt = task.prepare_prompt(asyncio.get_running_loop().create_future(),
                                 failure_text='Red light violation', was_route_completed=completed)
    assert prompt.request.user_content == [
        TextBlock('FAILURE REPORT:\nRed light violation'),
        TextBlock(f'ROUTE STATUS: The route was {status}.'),
    ]


@pytest.mark.asyncio
async def test_description_task_ignores_unrequested_images_and_metadata(task_factory, simulation_data):
    task = task_factory({'descriptions': True})
    prompt = task.prepare_prompt(asyncio.get_running_loop().create_future(),
                                 descriptions='A red car is ahead.',
                                 images=[simulation_data.image_dir / 'unused.jpg'],
                                 frames=simulation_data.frames, failure_text='ignored')
    assert prompt.request.user_content == [TextBlock('A red car is ahead.', to_be_cached=True)]


@pytest.mark.parametrize('wants_images', [False, True])
def test_required_images_follow_task_flags(task_factory, tmp_path, wants_images):
    task = task_factory({'images': wants_images})
    images = [tmp_path / 'frame.jpg']
    assert task.get_required_image_paths(TaskData(images=images)) == (images if wants_images else None)


def test_missing_task_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match='Task config file not found'):
        Task(tmp_path / 'missing.yaml')


def test_missing_prompt_file_raises(task_factory, tmp_path):
    with pytest.raises(FileNotFoundError, match='System prompt file not found'):
        task_factory(system_prompt_path=str(tmp_path / 'missing.txt'))


def test_missing_prompt_setting_raises(task_factory):
    with pytest.raises(ValueError, match='system_prompt_path'):
        task_factory(system_prompt_path='')
