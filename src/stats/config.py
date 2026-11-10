"""Portable analysis profiles; paths are relative to the profile, never cwd."""

from copy import deepcopy
from pathlib import Path
import math

import yaml


def load_config(path, data_dir=None):
    """Read a YAML profile and resolve its input directory.

    Returned configuration is an ordinary dictionary, suitable for copying and
    passing to calculation functions. Copy a profile to adapt input files,
    experimental design, sampling parameters, and timing assumptions.
    """
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Analysis configuration must be a YAML mapping")

    # Resolve input locations once, so changing the shell/notebook directory does
    # not change which study data are read. Leave the caller's profile untouched.
    config = deepcopy(config)
    config["config_path"] = str(path)
    config["data_dir"] = str(
        Path(data_dir).resolve()
        if data_dir
        else (path.parent / config.get("data_dir", "../../data")).resolve()
    )

    # Preserve the declared scenario order: progression and modeled baseline
    # allocation depend on it, even though aggregate means do not.
    scenarios = config.get("scenarios", [])
    if (
        not scenarios
        or any(type(s) is not int or s < 0 for s in scenarios)
        or len(set(scenarios)) != len(scenarios)
    ):
        raise ValueError("scenarios must be distinct nonnegative integer IDs")
    if not config.get("selected_model") or not config.get("results"):
        raise ValueError("selected_model and results must be specified")
    for key in (
        "repetitions",
        "runs_per_session",
        "baseline_repetitions",
        "baseline_max_runs",
    ):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")

    # Fixed seeds make conditional subsampling and the paired bootstrap
    # reproducible. Neither operation should inherit global random state.
    sampling = config.get("sampling", {})
    for key in ("draws", "seed", "scene_seed"):
        if type(sampling.get(key)) is not int or sampling[key] < (
            2 if key == "draws" else 0
        ):
            raise ValueError(f"sampling.{key} must be a valid integer")

    # Generation has one measured duration per feedback stage. Worker counts
    # may change in sensitivities, but these durations remain model inputs.
    timing = config.get("timing", {})
    durations = timing.get("generation_seconds", [])
    if len(durations) != config["runs_per_session"]:
        raise ValueError(
            "timing.generation_seconds must have one duration per session run"
        )
    for key, value in [("generation_seconds", v) for v in durations] + [
        (k, timing.get(k))
        for k in (
            "simulation_seconds",
            "baseline_generation_seconds",
            "batch_seconds",
            "batch_spread",
        )
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key in {"generation_seconds", "simulation_seconds"} and value == 0)
        ):
            raise ValueError(f"timing.{key} must be a finite valid duration")
    if (
        type(timing.get("baseline_generation_attempts")) is not int
        or timing["baseline_generation_attempts"] < 1
    ):
        raise ValueError("timing.baseline_generation_attempts must be positive")
    return config


def input_path(config, filename):
    return Path(config["data_dir"]) / filename


def selected_results(frame, config):
    """Baseline and the selected generator's two variants, for RQ2–RQ4."""
    result = frame.loc[frame.model.isin(["baseline", config["selected_model"]])].copy()
    expected = {
        "baseline / rule-based",
        *(f"{config['selected_model']} / {m}" for m in ("one-shot", "incremental")),
    }
    if set(result.setting) != expected:
        raise ValueError(
            "Selected analyses require baseline and both selected-model variants"
        )
    return result


def comparison_pairs(config):
    baseline = "baseline / rule-based"
    one = f"{config['selected_model']} / one-shot"
    incremental = f"{config['selected_model']} / incremental"
    return [(one, baseline), (incremental, baseline), (incremental, one)]
