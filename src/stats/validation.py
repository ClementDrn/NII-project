"""Validate exported data before computing any statistical quantity."""

import numpy as np
import pandas as pd

from .adversarial_analysis import KEY, load_csv
from .config import input_path


def load_results(config):
    """Load one CSV per configuration and enforce the declared design."""
    # Validate each CSV first, then check that separate files do not reuse an
    # experimental key. Duplicate texts are valid; duplicate experiments are not.
    frames = [
        load_csv(input_path(config, name), config["scenarios"])
        for name in config["results"]
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate experimental keys across result inputs")

    # Every repetition must cover the benchmark. Baseline may have fewer
    # available categories; generated methods require complete feedback sessions.
    for setting, group in frame.groupby("setting"):
        baseline = group.model.iloc[0] == "baseline"
        count = config["baseline_repetitions"] if baseline else config["repetitions"]
        if set(group.repetition) != set(range(count)):
            raise ValueError(f"{setting}: expected repetition IDs 0–{count - 1}")
        for repetition, repeated in group.groupby("repetition"):
            if set(repeated.scenario) != set(config["scenarios"]):
                raise ValueError(
                    f"{setting}, repetition {repetition}: incomplete scenario grid"
                )
            for scenario, session in repeated.groupby("scenario"):
                runs = set(session.run)
                allowed = set(
                    range(
                        config["baseline_max_runs"]
                        if baseline
                        else config["runs_per_session"]
                    )
                )
                if (not runs.issubset(allowed)) or (not baseline and runs != allowed):
                    raise ValueError(
                        f"{setting}, repetition {repetition}, scenario {scenario}: invalid run grid"
                    )
    return frame


def numeric_columns(frame, columns, label, lower=0, upper=None):
    """Return complete integer-valued measurements; never truncate or impute."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")

    # Reject fractions and missing measurements before any integer conversion.
    # Rounding or filling here would silently change the recorded observations.
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    array = values.to_numpy(dtype=float)
    if (
        not np.isfinite(array).all()
        or (array % 1 != 0).any()
        or (array < lower).any()
        or (upper is not None and (array > upper).any())
    ):
        raise ValueError(
            f"{label}: measurements must be complete finite integers in range"
        )
    return values
