"""Item-level human/LLM estimates and scenario-balanced full-pool ratings."""

import numpy as np
import pandas as pd
from scipy import stats

from .adversarial_analysis import paired_mean_comparisons
from .config import comparison_pairs, input_path, selected_results
from .validation import numeric_columns


def load_sample(config):
    """Join the explicit sampled-item methods to ratings by ID, retaining texts."""
    spec = config["naturalness"]
    items = pd.read_csv(input_path(config, spec["instructions"]), encoding="utf-8-sig")
    if not {"instruction_id", "instruction", "method"}.issubset(items):
        raise ValueError(
            "Sampled items require instruction_id, instruction, and method"
        )

    # Explicit item IDs and method labels define the sampling units. Inferring
    # the method from text membership would be ambiguous for duplicate texts.
    numeric_columns(items, ["instruction_id"], "sampled items")
    if (
        items.instruction_id.duplicated().any()
        or items[["instruction", "method"]].isna().any().any()
    ):
        raise ValueError(
            "Sampled items require unique IDs and complete text/method labels"
        )
    if items.instruction.str.strip().eq("").any():
        raise ValueError("Sampled instructions cannot be blank")
    if set(items.method) != {"baseline", "digit"}:
        raise ValueError("Sampled item methods must be baseline or digit (incremental)")
    if not items.groupby("method").size().eq(spec["sample_per_method"]).all():
        raise ValueError("Unexpected sampled-item counts per method")

    # Require the same complete item set for every evaluator. Checking row
    # counts alone would miss a duplicated item replacing an unrated one.
    expected = set(items.instruction_id)
    panels = []
    for label, filename in [("human", spec["human"]), ("LLM", spec["llm"])]:
        panel = pd.read_csv(input_path(config, filename), encoding="utf-8-sig")
        if not {"instruction_id", "user_id", "rating"}.issubset(panel):
            raise ValueError(f"{label}: expected instruction_id, user_id, rating")
        numeric_columns(panel, ["instruction_id"], label)
        numeric_columns(panel, ["rating"], label, lower=1, upper=5)
        if (
            panel.empty
            or panel.user_id.isna().any()
            or panel.user_id.astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"{label}: missing rater IDs or empty panel")
        panel["user_id"] = panel.user_id.astype(str)
        if panel.duplicated(["instruction_id", "user_id"]).any():
            raise ValueError(f"{label}: duplicate item/rater keys")
        if set(panel.instruction_id) != expected:
            raise ValueError(f"{label}: unknown or missing instruction IDs")
        for rater, ratings in panel.groupby("user_id"):
            if set(ratings.instruction_id) != expected:
                raise ValueError(f"{label}: rater {rater} has an incomplete panel")
        if label == "LLM" and (
            panel.user_id.nunique() != 1 or panel.instruction_id.duplicated().any()
        ):
            raise ValueError("Expected exactly one LLM rating per item")
        if label == "human" and panel.user_id.nunique() != spec["human_raters"]:
            raise ValueError("Unexpected number of complete human raters")
        panels.append(panel)

    # Average the human panel at the item level before estimating uncertainty.
    # The 11 ratings of one instruction are not 11 independently sampled items.
    human, llm = panels
    means = human.groupby("instruction_id").rating.mean().rename("human_mean")
    scores = items.merge(means, on="instruction_id", validate="one_to_one")
    scores = scores.merge(
        llm[["instruction_id", "rating"]].rename(columns={"rating": "llm_score"}),
        on="instruction_id",
        validate="one_to_one",
    )

    # Form paired judge-minus-panel discrepancies before taking variances;
    # separate judge/panel variances would lose their item-level covariance.
    scores["bias"] = scores.llm_score - scores.human_mean
    return scores, human


def sampled_estimates(scores, populations):
    """Welch–Satterthwaite t intervals with finite-population correction.

    Item means condition on the observed evaluator panel. Paired LLM-minus-human
    item differences retain covariance. No individual-rating iid assumption.
    """
    estimates, rows = {}, []

    def interval(mean, components):
        variance = sum(v for v, _ in components)
        if variance == 0:
            return mean, mean

        # Welch–Satterthwaite combines independently sampled method pools,
        # allowing their estimated sampling variances to differ.
        degrees = variance**2 / sum(v**2 / df for v, df in components)
        half = stats.t.ppf(0.975, degrees) * np.sqrt(variance)
        return mean - half, mean + half

    # First estimate each method in its own finite sampling population.
    for method, label in [("baseline", "baseline"), ("digit", "ours")]:
        group = scores.loc[scores.method.eq(method)]
        n, population = len(group), populations[method]
        if type(population) is not int or n < 2 or population < n:
            raise ValueError(
                "Population sizes must be integers at least the sampled count"
            )
        for quantity in ("human_mean", "llm_score", "bias"):
            values = group[quantity]
            # Sampling without replacement reduces uncertainty by 1 - n/N.
            # This still conditions on the observed evaluator panel and judge.
            mean, variance = (
                values.mean(),
                (1 - n / population) * values.var(ddof=1) / n,
            )
            estimates[method, quantity] = mean, variance, n
            low, high = interval(mean, [(variance, n - 1)])
            rows.append(
                dict(
                    quantity=f"{label}_{quantity}",
                    estimate=mean,
                    CI_95_low=low,
                    CI_95_high=high,
                )
            )

    # Between-method contrasts add the two sampling-variance contributions.
    # For comparison_error_E, those contributions come from paired item biases.
    for quantity, label in [
        ("human_mean", "human_difference"),
        ("llm_score", "llm_difference"),
        ("bias", "comparison_error_E"),
    ]:
        baseline_mean, baseline_variance, baseline_count = estimates[
            "baseline", quantity
        ]
        digit_mean, digit_variance, digit_count = estimates["digit", quantity]

        difference = digit_mean - baseline_mean
        variance_components = [
            (baseline_variance, baseline_count - 1),
            (digit_variance, digit_count - 1),
        ]
        low, high = interval(difference, variance_components)

        rows.append(
            dict(quantity=label, estimate=difference, CI_95_low=low, CI_95_high=high)
        )
    return pd.DataFrame(rows).set_index("quantity")


def selected_examples(scores):
    """Deterministic extrema and lower-middle item; ties use the lowest ID."""
    rows = []
    for method in ("digit", "baseline"):
        group = scores.loc[scores.method.eq(method)].sort_values(
            ["human_mean", "instruction_id"]
        )
        # Sorting by score then ID makes tied selections deterministic. With
        # an even sample, the lower central item need not attain the median.
        highest = group.loc[group.human_mean.eq(group.human_mean.max())].iloc[0]
        for label, row in [
            ("Highest", highest),
            ("Median", group.iloc[(len(group) - 1) // 2]),
            ("Lowest", group.iloc[0]),
        ]:
            rows.append(
                dict(
                    method=method,
                    selection=label,
                    instruction_id=row.instruction_id,
                    human_mean=row.human_mean,
                    llm_score=row.llm_score,
                    instruction=row.instruction,
                )
            )
    return pd.DataFrame(rows)


def analyze_naturalness(frame, config):
    """Return numerical tables and validated full-pool records for plotting."""
    scores, human = load_sample(config)
    report = sampled_estimates(scores, config["naturalness"]["populations"])
    full = selected_results(frame, config)
    full["naturalness"] = numeric_columns(
        full, ["naturalness"], "full-pool naturalness", 1, 5
    )
    if not full.usable.all():
        raise ValueError(
            "Full-pool naturalness requires scored usable instructions; review changed inputs"
        )

    # Full-pool judge ratings answer a separate question from the human sample:
    # give every benchmark scenario equal weight despite unequal pool sizes.
    scenario = full.groupby(["setting", "scenario"], as_index=False).naturalness.mean()
    summary = scenario.groupby("setting").naturalness.agg(["mean", "count"])
    contrasts = paired_mean_comparisons(
        scenario, "naturalness", comparison_pairs(config)
    )
    repetitions = (
        full.groupby(["setting", "repetition", "scenario"])
        .naturalness.mean()
        .groupby(["setting", "repetition"])
        .mean()
        .rename("mean")
        .reset_index()
    )

    # Plot labels use a common baseline method name; original CSV labels remain
    # untouched because selected_results returned a copy.
    full.loc[full.model.eq("baseline"), "method"] = "baseline"
    tables = {
        "naturalness_sample_items": scores,
        "naturalness_sample_estimates": report,
        "table_III": selected_examples(scores),
        "naturalness_full_summary": summary,
        "naturalness_scenario_means": scenario,
        "naturalness_full_contrasts": contrasts,
        "naturalness_repetition_means": repetitions,
        "naturalness_panel_audit": human.groupby("user_id")
        .size()
        .rename("rated_items")
        .to_frame(),
    }
    return tables, full
