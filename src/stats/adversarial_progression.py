"""Finite-pool failure accumulation at matched scenario allocation.

DIGIT replays repetition/scenario/run order. Baseline pools its two repetitions
within each scenario and uniformly samples records without replacement, taking
up to three per scenario per pass. Its curves are exact expectations over all
such shuffles, conditional on the recorded outcomes, not confidence intervals.
"""

from math import comb

import numpy as np
import pandas as pd

from .adversarial_analysis import BASELINE


def failure_progression(
    frame,
    scenarios,
    model="claude-opus-4.6",
    repetitions=3,
    runs_per_session=3,
    baseline_repetitions=2,
    baseline_max_runs=16,
):
    """Return cumulative curves and baseline allocation, including the origin.

    At every attempt through DIGIT's completion the scenario allocation matches
    its three-attempt sessions. Baseline repetition labels are retained in the
    input but pooled for sampling; passes are not experimental repetitions.
    Exhausted pools are skipped, with partial final blocks retained.
    """
    scenarios = list(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide distinct scenarios in the desired order")
    settings = [BASELINE, f"{model} / one-shot", f"{model} / incremental"]
    selected = frame.loc[frame.setting.isin(settings)].copy()
    keys = ["setting", "repetition", "scenario", "run"]
    if selected[keys].isna().any().any() or selected.duplicated(keys).any():
        raise ValueError("Missing or duplicate experimental keys")
    for column in ("repetition", "scenario", "run"):
        values = pd.to_numeric(selected[column], errors="raise")
        if (
            not np.isfinite(values).all()
            or (values % 1 != 0).any()
            or (values < 0).any()
        ):
            raise ValueError(f"{column} must contain nonnegative integer IDs")
        selected[column] = values.astype(int)
    if selected.adversarial.isna().any() or not selected.adversarial.isin([0, 1]).all():
        raise ValueError("adversarial must contain only 0/1 outcomes")
    for setting in settings:
        group = selected.loc[selected.setting.eq(setting)]
        if set(group.scenario) != set(scenarios):
            raise ValueError(f"Incomplete scenario coverage for {setting}")
        expected_reps = set(
            range(baseline_repetitions if setting == BASELINE else repetitions)
        )
        if set(group.repetition) != expected_reps:
            raise ValueError(f"Unexpected repetitions for {setting}")
        for repetition in expected_reps:
            repeated = group.loc[group.repetition.eq(repetition)]
            if set(repeated.scenario) != set(scenarios):
                raise ValueError(f"Incomplete repetition/scenario grid for {setting}")
            if setting == BASELINE:
                if not repeated.run.between(0, baseline_max_runs - 1).all():
                    raise ValueError("Baseline run IDs exceed the configured range")
            else:
                for _, session in repeated.groupby("scenario"):
                    if set(session.run) != set(range(runs_per_session)):
                        raise ValueError(
                            f"Incomplete three-attempt session for {setting}"
                        )

    baseline = selected.loc[selected.setting.eq(BASELINE)]
    pools = baseline.groupby("scenario").adversarial.agg(["size", "sum"])
    if pools["size"].min() < repetitions * runs_per_session:
        raise ValueError("Baseline needs enough records for the matched attempt budget")
    # Expectations at each possible within-scenario sample size. comb returns
    # zero when k exceeds the number of non-failures, including pool exhaustion.
    expectation = {}
    for scenario in scenarios:
        size, failures = (int(pools.loc[scenario, c]) for c in ("size", "sum"))
        expectation[scenario] = [
            (k * failures / size, 1 - comb(size - failures, k) / comb(size, k))
            for k in range(size + 1)
        ]

    def point(setting, attempts, failures, discovered, kind):
        return dict(
            setting=setting,
            attempts=attempts,
            cumulative_failures=failures,
            discovered_scenarios=discovered,
            curve_type=kind,
        )

    # Walk the baseline allocation one attempt at a time. Each point sums
    # scenario expectations at the current within-scenario budgets; no sampled
    # trajectory or graphical smoothing is involved.
    curves = [point(BASELINE, 0, 0.0, 0.0, "exact expectation")]
    allocation = []
    counts = dict.fromkeys(scenarios, 0)
    passes = 0
    while len(allocation) < len(baseline):
        for scenario in scenarios:
            remaining = int(pools.loc[scenario, "size"]) - counts[scenario]
            for _ in range(min(runs_per_session, remaining)):
                counts[scenario] += 1
                attempts = len(allocation) + 1
                allocation.append(
                    dict(
                        attempts=attempts,
                        scenario=scenario,
                        scenario_attempt=counts[scenario],
                        allocation_pass=passes,
                    )
                )
                expected = [expectation[s][counts[s]] for s in scenarios]
                failures, discovered = np.sum(expected, axis=0)
                curves.append(
                    point(BASELINE, attempts, failures, discovered, "exact expectation")
                )
        passes += 1

    # DIGIT is an observed replay, not a random subset. Keep each feedback
    # session intact and use the same scenario order as baseline allocation.
    rank = {scenario: index for index, scenario in enumerate(scenarios)}
    for setting in settings[1:]:
        target = selected.loc[selected.setting.eq(setting)].copy()
        target["scenario_order"] = target.scenario.map(rank)
        target = target.sort_values(["repetition", "scenario_order", "run"])
        curves.append(point(setting, 0, 0, 0, "observed ordered replay"))
        discovered, failures = set(), 0
        for attempts, row in enumerate(target.itertuples(), start=1):
            failures += int(row.adversarial)
            if row.adversarial:
                discovered.add(row.scenario)
            curves.append(
                point(
                    setting,
                    attempts,
                    failures,
                    len(discovered),
                    "observed ordered replay",
                )
            )
    return pd.DataFrame(curves), pd.DataFrame(allocation)
