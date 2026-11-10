"""Conditional attempt/time-matched discovery using observed outcomes.

Time comparisons replay existing outcomes under deterministic schedules. They
assume scheduling does not change outcomes or adaptive feedback. They are NOT
measured end-to-end experiments, and cannot extrapolate past observed pools.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .adversarial_analysis import BASELINE
from .pipeline_timing import simulate_pipeline

SIM_SECONDS = 29297 / 50
GEN_SECONDS = (6907 / 25, 20993 / 25, 24997 / 25)
BASELINE_GENERATION_ATTEMPTS = 800
BASELINE_GENERATION_SECONDS = 14.0


def campaign_schedule(
    scenarios,
    repetitions=3,
    mode="pipeline",
    workers=1,
    joint_repetitions=False,
    batch_first=False,
    batch_seconds=403.0,
    batch_spread=0.0,
    generation_seconds=GEN_SECONDS,
    simulation_seconds=SIM_SECONDS,
):
    """One simulator; return completion schedule for actual experimental keys.

    Default repetitions are consecutive campaigns, preserving the 25-scenario
    scope of the generation measurements. 'joint' launches all chains together.
    One-shot assumes each batch supplies all three instructions per scenario.
    Batched delivery is an explicit hypothesis, not inferred from first latency.
    """
    scenarios = list(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide distinct scenarios in the desired scheduling order")
    if repetitions < 1 or batch_seconds < 0 or batch_spread < 0:
        raise ValueError("Invalid campaign size or batch timing")
    if mode not in {"pipeline", "sequential", "one-shot"}:
        raise ValueError("Unknown scheduling mode")
    if joint_repetitions and mode != "pipeline":
        raise ValueError("Joint repetitions are implemented only for pipeline mode")
    rows, offset = [], 0.0
    n = len(scenarios)
    # Consecutive repetitions restart the queues after each campaign. Joint
    # scheduling is a separate sensitivity, not an interchangeable default.
    if mode == "pipeline":
        groups = (
            [list(range(repetitions))]
            if joint_repetitions
            else [[r] for r in range(repetitions)]
        )
        for reps in groups:
            summary, tasks = simulate_pipeline(
                scenarios=n,
                repetitions=len(reps),
                generation_seconds=generation_seconds,
                simulation_seconds=simulation_seconds,
                generation_workers=workers,
                first_batch_seconds=batch_seconds if batch_first else None,
                first_batch_spread=batch_spread if batch_first else 0.0,
            )

            # Record completed simulations with their original experimental
            # keys, so outcomes can later be replayed without resampling them.
            for task in tasks:
                if task["kind"] == "simulation":
                    rows.append(
                        {
                            "scenario": scenarios[task["scenario"]],
                            "repetition": reps[task["repetition"]],
                            "run": task["iteration"],
                            "completion_seconds": offset + task["end_seconds"],
                        }
                    )
            offset += summary["makespan_seconds"]
    else:
        for rep in range(repetitions):
            rep_start = offset
            for iteration in range(len(generation_seconds)):
                if mode == "sequential":
                    # Original non-overlapped reference: all G_i, then all S_i.
                    offset += n * generation_seconds[iteration]
                for index, scenario in enumerate(scenarios):
                    if mode == "one-shot":
                        ready = (
                            rep_start
                            + batch_seconds
                            + batch_spread * index / max(1, n - 1)
                        )
                        offset = max(offset, ready)
                    offset += simulation_seconds
                    rows.append(
                        {
                            "scenario": scenario,
                            "repetition": rep,
                            "run": iteration,
                            "completion_seconds": offset,
                        }
                    )
    return pd.DataFrame(rows).sort_values("completion_seconds").reset_index(drop=True)


def timing_schedules(
    scenarios,
    model="claude-opus-4.6",
    repetitions=3,
    batch_seconds=403.0,
    batch_spread=0.0,
    generation_seconds=GEN_SECONDS,
    simulation_seconds=SIM_SECONDS,
    baseline_generation_seconds=BASELINE_GENERATION_SECONDS,
    include_sensitivities=True,
):
    """Timing table and named schedules for paper and sensitivity cases."""
    configs = [
        ("one_shot_batch", "one-shot", dict(mode="one-shot")),
        ("incremental_sequential", "incremental", dict(mode="sequential")),
        ("incremental_pipeline_1", "incremental", dict(workers=1)),
        ("incremental_pipeline_2", "incremental", dict(workers=2)),
        ("incremental_pipeline_25", "incremental", dict(workers=25)),
        ("incremental_batch_first_1", "incremental", dict(workers=1, batch_first=True)),
        (
            "incremental_joint_pipeline_1",
            "incremental",
            dict(workers=1, joint_repetitions=True),
        ),
        (
            "incremental_joint_pipeline_2",
            "incremental",
            dict(workers=2, joint_repetitions=True),
        ),
        (
            "incremental_joint_batch_first_1",
            "incremental",
            dict(workers=1, joint_repetitions=True, batch_first=True),
        ),
    ]

    # Table II contains only these first four target schedules. Other queue
    # assumptions remain available through the diagnostics workflow.
    if not include_sensitivities:
        configs = configs[:4]
    schedules, rows = {}, []
    attempts = len(scenarios) * repetitions * len(generation_seconds)
    rows.append(
        {
            "schedule": "baseline",
            "setting": BASELINE,
            "attempts": attempts,
            "hours": (baseline_generation_seconds + attempts * simulation_seconds)
            / 3600,
            "repetition_scheduling": "one precomputed pool",
            "hypothetical_batch_delivery": False,
        }
    )
    for name, method, config in configs:
        schedule = campaign_schedule(
            scenarios,
            repetitions,
            batch_seconds=batch_seconds,
            batch_spread=batch_spread,
            generation_seconds=generation_seconds,
            simulation_seconds=simulation_seconds,
            **config,
        )
        setting = f"{model} / {method}"
        schedules[name] = (setting, schedule)
        rows.append(
            {
                "schedule": name,
                "setting": setting,
                "attempts": len(schedule),
                "hours": schedule.completion_seconds.max() / 3600,
                "repetition_scheduling": (
                    "joint" if config.get("joint_repetitions") else "consecutive"
                ),
                "hypothetical_batch_delivery": method == "one-shot"
                or config.get("batch_first", False),
            }
        )
    return pd.DataFrame(rows), schedules


def baseline_budget_counts(
    frame,
    scenarios,
    deadline_seconds,
    simulation_seconds=SIM_SECONDS,
    baseline_generation_seconds=BASELINE_GENERATION_SECONDS,
):
    """Round-robin scenarios; random order within each finite baseline pool.

    The full 800-attempt pool-generation cost is paid once, even if only a
    subset of the 658 available simulation records can be evaluated by T.
    """
    scenarios = list(scenarios)
    base = frame[frame.setting == BASELINE]
    capacity = base.groupby("scenario").size().reindex(scenarios)
    if capacity.isna().any() or set(base.scenario) != set(scenarios):
        raise ValueError("Baseline must cover exactly the selected scenarios")
    if not math.isfinite(deadline_seconds) or deadline_seconds < 0:
        raise ValueError("Deadline must be finite and nonnegative")

    # Subtract pool construction once. Only complete simulations count before
    # the deadline; the small tolerance absorbs floating-point boundary error.
    available = max(
        0,
        math.floor(
            (deadline_seconds - baseline_generation_seconds) / simulation_seconds + 1e-9
        ),
    )

    # Allocate round-robin in the fixed scenario order and skip exhausted pools.
    # This matches evaluation budgets without inventing unobserved records.
    counts = pd.Series(0, index=scenarios, dtype=int)
    for _ in range(int(capacity.max())):
        for scenario in scenarios:
            if available == 0:
                return counts
            if counts[scenario] < capacity[scenario]:
                counts[scenario] += 1
                available -= 1
    return counts


def conditional_novelty(
    frame, target, target_rows, baseline_counts, draws=20000, seed=20261020
):
    """Exact expected target-only fraction; conditional subset ranges via MC."""
    base = (
        frame[frame.setting == BASELINE]
        .groupby("scenario")
        .adversarial.agg(["sum", "size"])
    )
    if set(baseline_counts.index) != set(base.index):
        raise ValueError("Budget must specify every baseline scenario")
    k = baseline_counts.reindex(base.index).to_numpy(dtype=float)
    n, successes = base["size"].to_numpy(), base["sum"].to_numpy()
    if not np.isfinite(k).all() or (k % 1 != 0).any() or (k < 0).any() or (k > n).any():
        raise ValueError(
            "Baseline budgets must be integer counts within observed capacities"
        )
    k = k.astype(int)
    target_set = set(target_rows.loc[target_rows.adversarial == 1, "scenario"])
    a = np.array([s in target_set for s in base.index])

    # Hold target discoveries fixed. For each baseline scenario, p_none is the
    # exact probability that its allocated subset contains no observed failure.
    p_none = hypergeom.pmf(0, n, successes, k)

    # These draws describe variation among subsets of the existing pool. Their
    # quantiles are not confidence intervals for a population of new scenarios.
    rng = np.random.default_rng(seed)
    found = rng.hypergeometric(successes, n - successes, k, size=(draws, len(n))) > 0
    novelty_draws = (
        ((~found) & a).sum(axis=1) / a.sum() if a.any() else np.full(draws, np.nan)
    )
    return {
        "setting": target,
        "target_completed_attempts": len(target_rows),
        "baseline_completed_attempts": int(k.sum()),
        "baseline_min_per_scenario": int(k.min()),
        "baseline_max_per_scenario": int(k.max()),
        "target_failure_scenarios": int(a.sum()),
        "expected_baseline_failure_scenarios": float((1 - p_none).sum()),
        "expected_target_only_scenarios": float(p_none[a].sum()),
        "expected_baseline_only_scenarios": float((1 - p_none[~a]).sum()),
        "expected_novelty_fraction": float(p_none[a].mean()) if a.any() else np.nan,
        "novelty_subset_q025": (
            float(np.quantile(novelty_draws, 0.025)) if a.any() else np.nan
        ),
        "novelty_subset_q975": (
            float(np.quantile(novelty_draws, 0.975)) if a.any() else np.nan
        ),
    }


def budget_comparisons(
    frame,
    scenarios,
    schedules,
    common_deadline_hours,
    tries_per_scenario=9,
    draws=20000,
    seed=20261020,
    simulation_seconds=SIM_SECONDS,
    baseline_generation_seconds=BASELINE_GENERATION_SECONDS,
):
    """Same attempts, common-time horizon, and each target's completion horizon.

    Same-attempt comparisons use all observed nine attempts per target/scenario.
    For time comparisons, replay only attempts completed by the deadline. Never
    sample arbitrary incremental attempts out of their dependency chains.
    """
    keys = ["scenario", "repetition", "run"]
    results = []
    # Equal-attempt comparisons use the full recorded target campaign.
    for target in sorted({target for target, _ in schedules.values()}):
        target_rows = frame[frame.setting == target]
        counts = target_rows.groupby("scenario").size().reindex(scenarios)
        if not counts.eq(tries_per_scenario).all():
            raise ValueError(
                "Attempt comparison requires the specified number per scenario"
            )
        result = conditional_novelty(
            frame,
            target,
            target_rows,
            pd.Series(tries_per_scenario, index=scenarios),
            draws,
            seed,
        )
        results.append(
            {
                "comparison": "same_attempt_count",
                "schedule": "observed",
                "deadline_hours": np.nan,
                "target_pool_exhausted": True,
                **result,
            }
        )

    # Time comparisons instead censor the keyed replay at a deadline. Selecting
    # arbitrary target rows here would break the incremental feedback chains.
    for name, (target, schedule) in schedules.items():
        target_data = frame[frame.setting == target]
        if len(target_data) != len(schedule):
            raise ValueError(
                "Schedule must contain exactly the observed target attempts"
            )
        replay = schedule.merge(
            target_data, on=keys, how="left", validate="one_to_one", indicator=True
        )
        if not replay._merge.eq("both").all():
            raise ValueError("Schedule keys do not match observed outcome records")
        finish = schedule.completion_seconds.max()
        for label, deadline in [
            ("same_time_common_deadline", common_deadline_hours * 3600),
            ("same_time_target_completion", finish),
        ]:
            completed = replay[replay.completion_seconds <= deadline + 1e-8]
            base_counts = baseline_budget_counts(
                frame,
                scenarios,
                deadline,
                simulation_seconds,
                baseline_generation_seconds,
            )
            result = conditional_novelty(
                frame, target, completed, base_counts, draws, seed
            )
            results.append(
                {
                    "comparison": label,
                    "schedule": name,
                    "deadline_hours": deadline / 3600,
                    "target_pool_exhausted": bool(len(completed) == len(target_data)),
                    **result,
                }
            )
    return pd.DataFrame(results)
