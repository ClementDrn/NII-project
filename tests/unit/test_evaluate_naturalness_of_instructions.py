import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "evaluate_naturalness_of_instructions.py"
spec = importlib.util.spec_from_file_location("evaluate_naturalness", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def open_connection(self):
        pass

    async def prompt(self, request):
        self.prompts.append(request)
        return self.response

    async def stop(self):
        self.stopped = True


def test_parse_score_accepts_only_a_single_digit():
    assert module.parse_score(" 4 ") == "4"
    assert module.parse_score("4\nExplanation") == "N/A"
    assert module.parse_score("6") == "N/A"
    assert module.parse_score("") == "N/A"


def test_parse_batch_scores_maps_by_number_and_marks_invalid_rows():
    response = "3: 5\n1: 2\n2: 9\n3: 4\nextra text"

    assert module.parse_batch_scores(response, 3) == ["2", "N/A", "5"]


def test_build_batch_prompt_numbers_every_instruction():
    prompt = module.build_batch_prompt(["First command", "Second command"])

    assert "1. First command" in prompt
    assert "2. Second command" in prompt
    assert "number: digit" in prompt


@pytest.mark.asyncio
async def test_single_request_mode_makes_one_model_request(monkeypatch):
    model = FakeModel("1: 4\n2: 5")
    monkeypatch.setattr(module, "_create_model", lambda: model)
    rows = [{"instruction": "First command"}, {"instruction": "Second command"}]

    assert await module.score_rows(rows, single_request=True) == ["4", "5"]
    assert len(model.prompts) == 1
    assert model.started
    assert model.stopped


@pytest.mark.asyncio
async def test_default_mode_makes_one_request_per_row(monkeypatch):
    model = FakeModel("3")
    monkeypatch.setattr(module, "_create_model", lambda: model)
    rows = [{"instruction": "First command"}, {"instruction": "Second command"}]

    assert await module.score_rows(rows, single_request=False) == ["3", "3"]
    assert len(model.prompts) == 2
