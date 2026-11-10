"""Scenario-balanced scene mentions, paired cluster bootstrap, and object types.

Counts are precomputed annotations, not fresh NLP evaluations. Repeated text
retains its observed frequency. Flagged mentions are subsets of total mentions.
"""

import numpy as np
import pandas as pd

from .adversarial_analysis import BASELINE
from .config import selected_results, comparison_pairs


def analyze_scene(frame, config):
    """Return numerical scene-reference and object-type tables."""
    df = selected_results(frame, config)
    all_scenarios = config["scenarios"]
    ONE_SHOT = f"{config['selected_model']} / one-shot"
    INCREMENTAL = f"{config['selected_model']} / incremental"
    COMPARISONS = comparison_pairs(config)
    SCENE_METRICS = [
        "object_count",
        "modifier_count",
        "object_hallucination_count",
        "modifier_hallucination_count",
    ]
    scene_order = [BASELINE, ONE_SHOT, INCREMENTAL]

    # Scene quality is conditional on usable instructions. Record exclusions
    # explicitly, while end-to-end effectiveness retains those same attempts.
    scene_df = df.loc[df["usable"]].copy()
    scene_audit = df.groupby("setting").agg(
        attempts=("usable", "size"), usable=("usable", "sum")
    )
    scene_audit["excluded"] = scene_audit["attempts"] - scene_audit["usable"]

    # Missing annotations are missing measurements, not zero mentions. Check
    # the whole usable set before computing any scenario average.
    scene_counts = scene_df[SCENE_METRICS].apply(pd.to_numeric, errors="raise")
    scene_audit["missing_count_rows"] = (
        scene_counts.isna().any(axis=1).groupby(scene_df["setting"]).sum()
    )
    if (
        not np.isfinite(scene_counts.to_numpy()).all()
        or (scene_counts < 0).any().any()
        or (scene_counts % 1 != 0).any().any()
    ):
        raise ValueError("Scene counts must be complete, finite, nonnegative integers")
    scene_df[SCENE_METRICS] = scene_counts

    # Flagged mentions are contained within totals. Subtract them to obtain
    # unflagged counts; do not add them as an extra category above the total.
    for noun in ["object", "modifier"]:
        total, hallucinated = f"{noun}_count", f"{noun}_hallucination_count"
        if (scene_df[hallucinated] > scene_df[total]).any():
            raise ValueError(f"Hallucinated {noun} counts exceed total mentions")
        scene_df[f"{noun}_nonflagged_count"] = scene_df[total] - scene_df[hallucinated]
        scene_df[f"any_{noun}"] = scene_df[total].gt(0).astype(float)
    scene_df["any_hallucination"] = (
        scene_df[["object_hallucination_count", "modifier_hallucination_count"]]
        .sum(axis=1)
        .gt(0)
        .astype(float)
    )
    scene_extra = [
        "object_nonflagged_count",
        "modifier_nonflagged_count",
        "any_object",
        "any_modifier",
        "any_hallucination",
    ]

    # Average instructions inside each scenario first. This prevents baseline
    # scenarios with more evaluated instructions from dominating the result.
    scene_scenario = scene_df.groupby(["setting", "scenario"])[
        SCENE_METRICS + scene_extra
    ].mean()
    expected_scene_index = pd.MultiIndex.from_product(
        [scene_order, all_scenarios], names=["setting", "scenario"]
    )
    scene_scenario = scene_scenario.reindex(expected_scene_index)
    if scene_scenario.isna().any().any():
        raise ValueError(
            "Every method must have usable, scored instructions in every benchmark scenario"
        )

    # A flagged share is a ratio of balanced mean counts, not an average of
    # instruction-level percentages (which would weight sparse mentions oddly).
    scene_summary = scene_scenario.groupby(level="setting").mean().reindex(scene_order)
    for noun in ["object", "modifier"]:
        scene_summary[f"{noun}_hallucination_pct"] = (
            100
            * scene_summary[f"{noun}_hallucination_count"]
            / scene_summary[f"{noun}_count"].replace(0, np.nan)
        )
    scene_summary["modifiers_per_object"] = scene_summary[
        "modifier_count"
    ] / scene_summary["object_count"].replace(0, np.nan)
    scene_totals = scene_df.groupby("setting")[SCENE_METRICS].sum().reindex(scene_order)
    # Paired scenario-cluster bootstrap: retain dependencies within each scenario.
    SCENE_BOOTSTRAP_DRAWS = config["sampling"]["draws"]
    scene_rng = np.random.default_rng(config["sampling"]["scene_seed"])
    # Reuse these scenario indices for every method and metric. That pairing
    # preserves cross-method dependence when forming bootstrap contrasts.
    scene_draws = scene_rng.integers(
        0, len(all_scenarios), size=(SCENE_BOOTSTRAP_DRAWS, len(all_scenarios))
    )
    scene_boot = {}
    scene_ci_rows = []
    for setting in scene_order:
        values = scene_scenario.loc[setting].reindex(all_scenarios)
        boot_means = values.to_numpy()[scene_draws].mean(axis=1)
        samples = {metric: boot_means[:, j] for j, metric in enumerate(values.columns)}

        # Recalculate ratios within each bootstrap draw. A draw with zero total
        # mentions has an undefined share and is tracked via valid draw counts.
        for noun in ["object", "modifier"]:
            den = samples[f"{noun}_count"]
            samples[f"{noun}_hallucination_pct"] = np.divide(
                100 * samples[f"{noun}_hallucination_count"],
                den,
                out=np.full_like(den, np.nan),
                where=den > 0,
            )
        den = samples["object_count"]
        samples["modifiers_per_object"] = np.divide(
            samples["modifier_count"], den, out=np.full_like(den, np.nan), where=den > 0
        )
        scene_boot[setting] = samples
        for metric, sample in samples.items():
            low, high = np.nanquantile(sample, [0.025, 0.975])
            scene_ci_rows.append(
                dict(
                    setting=setting,
                    metric=metric,
                    mean=scene_summary.loc[setting, metric],
                    CI_95_low=low,
                    CI_95_high=high,
                    valid_bootstrap_draws=int(np.isfinite(sample).sum()),
                )
            )
    scene_ci = pd.DataFrame(scene_ci_rows).set_index(["setting", "metric"])

    # Subtract matched bootstrap draws, rather than separately estimated
    # interval endpoints, to obtain each paired method-difference interval.
    scene_contrast_rows = []
    for target, reference in COMPARISONS:
        for metric in scene_summary.columns:
            sample = scene_boot[target][metric] - scene_boot[reference][metric]
            low, high = np.nanquantile(sample, [0.025, 0.975])
            row = dict(
                target=target,
                reference=reference,
                metric=metric,
                mean_difference=scene_summary.loc[target, metric]
                - scene_summary.loc[reference, metric],
                CI_95_low=low,
                CI_95_high=high,
            )
            if metric in scene_scenario.columns:
                delta = (
                    scene_scenario.loc[target, metric]
                    - scene_scenario.loc[reference, metric]
                )
                row.update(
                    higher_scenarios=int((delta > 1e-12).sum()),
                    lower_scenarios=int((delta < -1e-12).sum()),
                    tied_scenarios=int((delta.abs() <= 1e-12).sum()),
                )
            scene_contrast_rows.append(row)
    scene_contrasts = pd.DataFrame(scene_contrast_rows)

    # Object types describe mention composition. Their row totals must agree
    # with the parser's object count before any frequencies are interpreted.
    OBJECT_TYPES = (
        "car motorbike bicycle pedestrian traffic_sign traffic_light road "
        "driving_lane parking_lane bike_lane shoulder sidewalk crosswalk "
        "intersection junction construction accident misc"
    ).split()
    object_type_order = [BASELINE, ONE_SHOT, INCREMENTAL]
    object_type_df = df.loc[df.usable].copy()
    # Retain the CSV's extra unknown category when present; do not silently omit it.
    object_type_columns = OBJECT_TYPES + (
        ["unknown"] if "unknown" in object_type_df else []
    )
    object_type_counts = object_type_df[object_type_columns].apply(
        pd.to_numeric, errors="raise"
    )
    if (
        not np.isfinite(object_type_counts.to_numpy()).all()
        or (object_type_counts < 0).any().any()
        or (object_type_counts % 1 != 0).any().any()
    ):
        raise ValueError("Object-type counts must be complete nonnegative integers")
    if (
        not object_type_counts.sum(axis=1)
        .eq(pd.to_numeric(object_type_df.object_count, errors="raise"))
        .all()
    ):
        raise ValueError(
            "Type counts do not sum to object_count; check category definitions before interpreting composition"
        )
    object_type_df[object_type_columns] = object_type_counts
    object_type_scenarios = object_type_df.groupby(["setting", "scenario"])[
        object_type_columns
    ].mean()
    object_type_scenarios = object_type_scenarios.reindex(
        pd.MultiIndex.from_product(
            [object_type_order, all_scenarios], names=["setting", "scenario"]
        )
    )
    if object_type_scenarios.isna().any().any():
        raise ValueError(
            "Every method must cover all scenarios with usable scored records"
        )

    # Scale balanced mention counts to mentions per 100 instructions. Values
    # are counts and can exceed 100; they are not instruction percentages.
    object_type_frequency = 100 * object_type_scenarios.groupby(
        level="setting"
    ).mean().reindex(object_type_order)
    object_type_share = (
        object_type_frequency.div(
            object_type_frequency.sum(axis=1).replace(0, np.nan), axis=0
        )
        * 100
    )
    reference_ratios = scene_summary[["object_count", "modifier_count"]].div(
        scene_summary.loc[BASELINE, ["object_count", "modifier_count"]].replace(
            0, np.nan
        )
    )
    reference_ratios.columns = [
        "object_ratio_to_baseline",
        "modifier_ratio_to_baseline",
    ]

    return {
        "scene_relevance_audit": scene_audit,
        "scene_relevance_summary": scene_summary,
        "scene_reference_ratios": reference_ratios,
        "scene_relevance_scenarios": scene_scenario,
        "scene_relevance_totals": scene_totals,
        "scene_relevance_contrasts": scene_contrasts,
        "scene_relevance_ci": scene_ci,
        "object_type_frequency": object_type_frequency,
        "object_type_share": object_type_share,
        "object_type_scenarios": object_type_scenarios,
        "object_type_counts": object_type_frequency.gt(0)
        .sum(axis=1)
        .rename("observed_types")
        .to_frame(),
    }
