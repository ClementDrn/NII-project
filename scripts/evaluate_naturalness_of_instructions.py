"""Score the naturalness of passenger instructions with gpt-5-mini."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT / "src"))

from untitled.core.base import CONFIG, LOG
from untitled.core.settings import (
	get_host_config_by_id,
	get_model_instance_config_by_id,
)
from untitled.generator.model_config import get_model_config
from untitled.generator.model_client import ModelType, parse_model_response
from untitled.generator.remote_model import RemoteModel
from untitled.generator.request import Request, TextBlock


NATURALNESS_COLUMN = "naturalness"
MODEL_INSTANCE_ID = "gpt-5-mini_0"
INVALID_SCORE = "N/A"

SCORING_PROMPT = """**Context:** A passenger is speaking aloud to a self-driving car.

**Question:** How natural does this spoken command sound for a human passenger to say?

- Focus solely on the phrasing and the words used (how natural it sounds for a human passenger to say).
- Do not evaluate whether the driving action is safe, legal, or polite.
- Your answer should solely be a digit between 1 and 5:
  - 1: not at all natural
  - 2: slightly natural
  - 3: moderately natural
  - 4: very natural
  - 5: extremely natural

Below is the command:
{instruction}
"""

BATCH_PROMPT = """**Context:** A passenger is speaking aloud to a self-driving car.

For each numbered command below, rate only how natural its phrasing sounds for a human
passenger to say. Do not evaluate whether the driving action is safe, legal, or polite.
Return exactly one line per command in the format `number: digit`, where the digit is
between 1 and 5. Do not include explanations or other text.

Scoring:
1 = not at all natural
2 = slightly natural
3 = moderately natural
4 = very natural
5 = extremely natural

Commands:
{commands}
"""


def build_prompt(instruction: str) -> str:
	return SCORING_PROMPT.format(instruction=instruction)


def build_batch_prompt(instructions: Iterable[str]) -> str:
	commands = "\n".join(
		f"{index}. {instruction}"
		for index, instruction in enumerate(instructions, start=1)
	)
	return BATCH_PROMPT.format(commands=commands)


def parse_score(response: str) -> str:
	"""Return a valid single score, or ``N/A`` for malformed model output."""
	match = re.fullmatch(r"\s*([1-5])\s*", parse_model_response(response))
	return match.group(1) if match else INVALID_SCORE


def parse_batch_scores(response: str, row_count: int) -> list[str]:
	"""Parse ``number: score`` lines and retain valid scores by row number."""
	scores = [INVALID_SCORE] * row_count
	seen: set[int] = set()
	pattern = re.compile(r"^\s*(\d+)\s*:\s*([1-5])\s*$")

	for line in parse_model_response(response).splitlines():
		match = pattern.fullmatch(line)
		if not match:
			continue
		number = int(match.group(1))
		if not 1 <= number <= row_count or number in seen:
			continue
		seen.add(number)
		scores[number - 1] = match.group(2)

	return scores


def _request(prompt: str) -> Request:
	return Request(system_text="", user_content=[TextBlock(text=prompt)])


def _create_model() -> RemoteModel:
	from untitled.generator.providers.litellm import LiteLLMAdapter

	instance = get_model_instance_config_by_id(CONFIG, MODEL_INSTANCE_ID)
	if instance.remote_config is None:
		raise ValueError(f"Model instance {MODEL_INSTANCE_ID!r} is not remote")

	host_config = get_host_config_by_id(CONFIG, instance.remote_config.host_id)
	model_config = get_model_config("gpt-5-mini")
	api_parameters = {
		"model": model_config.model_name,
		model_config.max_output_tokens_param: 128,
		**(model_config.thinking_deactivation_params or {}),
	}
	adapter = LiteLLMAdapter(
		host=host_config.host,
		api_key=host_config.api_key,
		api_parameters=api_parameters
    )
	return RemoteModel(
		model_type=ModelType.LLM,
		priority=instance.priority,
		batch_size=instance.batch_size,
		adapter=adapter,
	)


async def score_rows(rows: list[dict[str, str]], single_request: bool) -> list[str]:
	model = _create_model()
	await model.start()
	try:
		await model.open_connection()
		if single_request:
			response = await model.prompt(_request(build_batch_prompt(row["instruction"] for row in rows)))
			scores = parse_batch_scores(response, len(rows))
			if any(score == INVALID_SCORE for score in scores):
				LOG.warning("One or more batch naturalness scores were invalid or missing")
			return scores

		scores = []
		for index, row in enumerate(rows, start=1):
			response = await model.prompt(_request(build_prompt(row["instruction"])))
			score = parse_score(response)
			if score == INVALID_SCORE:
				LOG.warning("Invalid naturalness score for row %d: %r", index, response)
			scores.append(score)
		return scores
	finally:
		await model.stop()


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("input_csv", type=Path, help="Input CSV containing an instruction column")
	parser.add_argument("output_csv", type=Path, help="Output CSV to create")
	parser.add_argument(
		"--single-request",
		action="store_true",
		help="Send all instructions in one LLM request instead of one request per row",
	)
	return parser.parse_args()


def _load_rows(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
	with input_path.open("r", encoding="utf-8", newline="") as file:
		reader = csv.DictReader(file)
		if reader.fieldnames is None or "instruction" not in reader.fieldnames:
			raise ValueError("Input CSV must contain an 'instruction' column")
		return list(reader.fieldnames), list(reader)


async def _run(input_path: Path, output_path: Path, single_request: bool) -> None:
	if input_path.resolve() == output_path.resolve():
		raise ValueError("Input and output CSV paths must be different")
	if not input_path.is_file():
		raise FileNotFoundError(input_path)

	fieldnames, rows = _load_rows(input_path)
	scores = await score_rows(rows, single_request)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=[*fieldnames, NATURALNESS_COLUMN])
		writer.writeheader()
		for row, score in zip(rows, scores):
			writer.writerow({**row, NATURALNESS_COLUMN: score})


def main() -> None:
	logging.basicConfig(level=logging.INFO)
	args = _parse_args()
	asyncio.run(_run(args.input_csv, args.output_csv, args.single_request))


if __name__ == "__main__":
	main()
