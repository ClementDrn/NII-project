"""Analysis workflows returning numerical DataFrames, with no filesystem writes."""

import pandas as pd

from .adversarial_analysis import (
    BASELINE,
    digit_configuration_comparisons,
    matched_budget_overlap,
    model_summary_tables,
    observed_overlap,
    paired_mean_comparisons,
    scenario_table,
    session_summaries,
    within_method_overlap,
)
from .adversarial_progression import failure_progression
from .budget_novelty import baseline_budget_counts, budget_comparisons, timing_schedules
from .config import comparison_pairs, selected_results


def effectiveness(frame, config):
    """Summarize attempted-test outcomes and compare scenario-level rates."""
    # Keep all configurations in the main table. The selected-model subset is
    # used only for the paper's detailed baseline/variant contrasts.
    summary, publication, repetitions = model_summary_tables(frame, config["scenarios"])
    chosen = selected_results(frame, config)

    # Export both multiplicity choices; do not let the reporting layer pick a
    # different family merely because its significance decisions are favorable.
    comparisons = digit_configuration_comparisons(frame)
    inference = comparisons.groupby("metric").agg(
        joint_rejections=("p_t_Holm_all", lambda p: int(p.lt(0.05).sum())),
        endpoint_rejections=("p_t_Holm_endpoint", lambda p: int(p.lt(0.05).sum())),
    )

    return {
        "all_model_summary": summary,
        "table_I": publication.loc[publication.Model.ne("baseline")],
        "baseline_summary": publication.loc[publication.Model.eq("baseline")],
        "repetition_rates": repetitions,
        "scenario_rates": scenario_table(frame),
        "selected_effectiveness_contrasts": paired_mean_comparisons(
            scenario_table(chosen), "adversarial_rate", comparison_pairs(config)
        ),
        "configuration_comparisons": comparisons,
        "inference_summary": inference,
    }


def timing(config, sensitivities=False):
    """Build the paper schedules, optionally adding diagnostic assumptions."""
    # Durations are measured inputs to a deterministic scheduling model. This
    # function does not estimate speedups from the experiment CSV timestamps.
    spec = config["timing"]

    return timing_schedules(
        config["scenarios"],
        config["selected_model"],
        config["repetitions"],
        batch_seconds=spec["batch_seconds"],
        batch_spread=spec["batch_spread"],
        generation_seconds=spec["generation_seconds"],
        simulation_seconds=spec["simulation_seconds"],
        baseline_generation_seconds=spec["baseline_generation_seconds"],
        include_sensitivities=sensitivities,
    )


def complementarity(frame, config, sensitivities=False):
    """Compare discovered scenarios under explicit attempt and time budgets."""
    chosen = selected_results(frame, config)
    schedules, replay = timing(config, sensitivities)

    # Figure 5 uses one shared deadline: baseline's modeled completion time.
    # Table II instead gives baseline as much time as each target needs.
    common = float(schedules.loc[schedules.schedule.eq("baseline"), "hours"].iloc[0])
    sampling, spec = config["sampling"], config["timing"]

    # k is the total number of recorded target attempts per scenario, across
    # repetitions. It is not the number of distinct benchmark scenarios.
    k = config["repetitions"] * config["runs_per_session"]

    budgets = budget_comparisons(
        chosen,
        config["scenarios"],
        replay,
        common,
        tries_per_scenario=k,
        draws=sampling["draws"],
        seed=sampling["seed"],
        simulation_seconds=spec["simulation_seconds"],
        baseline_generation_seconds=spec["baseline_generation_seconds"],
    )

    # Progression uses three-attempt scenario blocks. Time-budget baseline
    # allocation uses round-robin attempts; keep the two allocations separate.
    progression, allocation = failure_progression(
        chosen,
        config["scenarios"],
        config["selected_model"],
        config["repetitions"],
        config["runs_per_session"],
        config["baseline_repetitions"],
        config["baseline_max_runs"],
    )

    paper = budgets.loc[budgets.comparison.eq("same_time_target_completion")].copy()
    paper["novelty_percent"] = 100 * paper.expected_novelty_fraction

    # The target count is observed; baseline-only and target-only regions are
    # conditional expectations over finite baseline subsets. Region arithmetic
    # is therefore allowed to produce fractional scenario counts.
    overlap = budgets.loc[
        budgets.comparison.eq("same_time_common_deadline")
        & budgets.schedule.isin(["one_shot_batch", "incremental_pipeline_1"])
    ].copy()
    overlap["shared"] = (
        overlap.target_failure_scenarios - overlap.expected_target_only_scenarios
    )
    overlap["neither"] = (
        len(config["scenarios"])
        - overlap.expected_baseline_only_scenarios
        - overlap.target_failure_scenarios
    )

    # Hold each observed target campaign fixed while rarefying baseline to k
    # attempts per scenario. This comparison matches attempts, not elapsed time.
    matched = pd.DataFrame(
        [
            matched_budget_overlap(
                chosen, target, k, sampling["draws"], sampling["seed"]
            )
            for target, _ in comparison_pairs(config)[:2]
        ]
    )

    # The paper's failure-count improvement uses the shared completion budget.
    # It is distinct from the scenario-balanced effectiveness-rate comparison.
    end = progression.loc[
        progression.attempts.eq(k * len(config["scenarios"]))
    ].set_index("setting")
    expected = end.loc[BASELINE, "cumulative_failures"]
    improvement = (
        end.loc[end.index != BASELINE, ["cumulative_failures"]] / expected - 1
    ) * 100
    improvement = improvement.rename(
        columns={"cumulative_failures": "failure_count_improvement_percent"}
    )

    # Save the per-scenario budgets behind every modeled comparison, allowing
    # readers to audit partial round-robin passes and finite-pool exhaustion.
    allocations = []
    for row in budgets.itertuples():
        if pd.notna(row.deadline_hours):
            counts = baseline_budget_counts(
                chosen,
                config["scenarios"],
                row.deadline_hours * 3600,
                spec["simulation_seconds"],
                spec["baseline_generation_seconds"],
            )
            allocations.extend(
                dict(
                    comparison=row.comparison,
                    schedule=row.schedule,
                    scenario=s,
                    attempts=int(n),
                )
                for s, n in counts.items()
            )

    return {
        "timing_summary": schedules,
        "table_II": paper,
        "novelty_budgets": budgets,
        "overlap_regions": overlap,
        "adversarial_progression": progression,
        "baseline_progression_allocation": allocation,
        "baseline_deadline_allocations": pd.DataFrame(allocations),
        "matched_budget_overlap": matched,
        "failure_count_improvement": improvement,
        "modeled_completion_schedule": pd.concat(
            [
                schedule.assign(schedule=name, setting=setting)
                for name, (setting, schedule) in replay.items()
            ],
            ignore_index=True,
        ),
    }


def diagnostics(frame, config):
    """Describe sessions and overlap without treating them as independent tests."""
    # Repetition-pair Jaccards share data, and session success is a different
    # quantity from per-attempt yield. Keep the supporting summaries separate.
    sessions, summary, repetitions, baseline_probability = session_summaries(
        frame, config["runs_per_session"]
    )

    return {
        "sessions": sessions,
        "session_summary": summary,
        "session_repetitions": repetitions,
        "baseline_matched_session": pd.DataFrame(
            {"discovery_probability": [baseline_probability]}
        ),
        "within_method_overlap": within_method_overlap(frame),
        "full_pool_overlap": observed_overlap(selected_results(frame, config)),
    }
