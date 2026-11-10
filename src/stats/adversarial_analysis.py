"""Validated, scenario-balanced summaries for DIGIT result CSVs.

Primary point estimates describe the observed benchmark. Paired t intervals/tests
are exploratory scenario-level inference, requiring independent, representative
scenarios for population generalization. Subsampling ranges are NOT confidence
intervals. No row-level independence assumption is used for scenario contrasts.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

KEY = ["model", "method", "repetition", "scenario", "run"]
BASELINE = "baseline / rule-based"


def parse_binary(series, column_name):
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{column_name} must contain only 0/1 values")
    return values.astype(int)


def load_csv(path, expected_scenarios):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = set(KEY) | {"redacted?", "errored?", "adversarial?", "instruction"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty or frame[KEY].isna().any().any():
        raise ValueError(f"{path}: empty data or missing experimental IDs")
    if len(frame[["model", "method"]].drop_duplicates()) != 1:
        raise ValueError(f"{path}: expected exactly one model/method setting")

    # IDs define experimental units. A fractional ID must fail rather than be
    # truncated into a different scenario, repetition, or run.
    for column in ["scenario", "repetition", "run"]:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(numeric).all() or ((numeric % 1 != 0) | (numeric < 0)).any():
            raise ValueError(f"{path}: {column} must contain finite integer IDs")
        frame[column] = numeric.astype(int)
    if frame.duplicated(KEY).any():
        raise ValueError(
            f"{path}: duplicate experimental keys; resolve provenance first"
        )
    if set(frame.scenario) != set(expected_scenarios):
        raise ValueError(f"{path}: scenario set does not match the benchmark")

    # Keep the CSV's raw outcome alongside the derived end-to-end outcome.
    # Refusals/errors stay in the denominator even if a raw flag says failure.
    for source, target in [
        ("redacted?", "redacted"),
        ("errored?", "errored"),
        ("adversarial?", "adversarial_raw"),
    ]:
        frame[target] = parse_binary(frame[source], source)
    frame["setting"] = frame["model"] + " / " + frame["method"]
    frame["usable"] = (frame.redacted == 0) & (frame.errored == 0)
    blank = frame.instruction.isna() | frame.instruction.fillna("").str.strip().eq("")
    if (blank & frame.usable).any():
        raise ValueError(f"{path}: usable runs have missing instruction text")
    # Observed end-to-end yield: unsuccessful generation/evaluation attempts
    # remain in the denominator. Preserve raw labels for auditing.
    frame["adversarial"] = frame.adversarial_raw * frame.usable.astype(int)
    return frame


def scenario_table(frame):
    # Start with one row per scenario and method. Later averages therefore
    # weight scenarios equally despite baseline's unequal numbers of attempts.
    table = frame.groupby(["setting", "scenario"], sort=True).agg(
        attempts=("adversarial", "size"),
        adversarial=("adversarial", "sum"),
        redacted=("redacted", "sum"),
        errors=("errored", "sum"),
        usable=("usable", "sum"),
    )
    table["adversarial_rate"] = table.adversarial / table.attempts
    table["redaction_rate"] = table.redacted / table.attempts
    table["error_rate"] = table.errors / table.attempts
    table["conditional_adversarial_rate"] = table.adversarial / table.usable.replace(
        0, np.nan
    )
    return table.reset_index()


def setting_summary(frame):
    per_scenario = scenario_table(frame)
    summary = per_scenario.groupby("setting").agg(
        attempts=("attempts", "sum"),
        adversarial=("adversarial", "sum"),
        redacted=("redacted", "sum"),
        errors=("errors", "sum"),
        scenarios=("scenario", "nunique"),
        mean_adversarial_rate=("adversarial_rate", "mean"),
        mean_redaction_rate=("redaction_rate", "mean"),
        mean_error_rate=("error_rate", "mean"),
        mean_conditional_adversarial_rate=("conditional_adversarial_rate", "mean"),
        scenarios_with_usable_output=("conditional_adversarial_rate", "count"),
        min_attempts_per_scenario=("attempts", "min"),
        max_attempts_per_scenario=("attempts", "max"),
    )

    # Export the pooled fraction separately; it answers a different question
    # from the equal-scenario mean when scenario allocations differ.
    summary["pooled_adversarial_rate"] = summary.adversarial / summary.attempts
    summary["failure_scenarios"] = per_scenario.groupby("setting").adversarial.apply(
        lambda x: (x > 0).sum()
    )
    return summary


def model_summary_tables(frame, expected_scenarios):
    """Return full summary, numeric publication table and repetition rates.

    Rates first average scenarios within each repetition, then repetitions.
    SD is the sample SD across repetitions, undefined for one campaign.
    Counts always sum all recorded attempts.
    """
    # Balance scenarios inside each repetition before calculating variability
    # across repetitions. Treating individual attempts as replicates would
    # ignore dependence within feedback sessions.
    per_scenario = frame.groupby(
        ["setting", "repetition", "scenario"], as_index=False
    ).agg(
        adversarial_rate=("adversarial", "mean"),
        redacted_rate=("redacted", "mean"),
    )
    for (setting, repetition), group in per_scenario.groupby(["setting", "repetition"]):
        if set(group.scenario) != set(expected_scenarios):
            raise ValueError(
                f"Incomplete scenario coverage: {setting}, repetition {repetition}"
            )

    # The paper's ± values describe repetition-to-repetition spread, not the
    # standard error of the overall mean. ddof=1 gives the sample SD.
    rates = per_scenario.groupby(["setting", "repetition"], as_index=False)[
        ["adversarial_rate", "redacted_rate"]
    ].mean()
    variability = rates.groupby("setting").agg(
        repetitions=("repetition", "size"),
        mean_repetition_rate=("adversarial_rate", "mean"),
        sd_repetition_rate=("adversarial_rate", lambda values: values.std(ddof=1)),
        mean_repetition_redacted_rate=("redacted_rate", "mean"),
        sd_repetition_redacted_rate=(
            "redacted_rate",
            lambda values: values.std(ddof=1),
        ),
    )
    summary = setting_summary(frame).join(variability, validate="one_to_one")
    table = (
        frame[["setting", "model", "method"]]
        .drop_duplicates()
        .merge(summary.reset_index(), on="setting", validate="one_to_one")
    )

    # Keep numerical columns separate from publication formatting. Scaling a
    # rate and its SD by 100 gives percent and percentage points respectively.
    table["Adversarial rate (%)"] = 100 * table.mean_repetition_rate
    table["Adversarial rate SD (pp)"] = 100 * table.sd_repetition_rate
    table["Redacted rate (%)"] = 100 * table.mean_repetition_redacted_rate
    table["Redacted rate SD (pp)"] = 100 * table.sd_repetition_redacted_rate
    names = {
        "model": "Model",
        "method": "Method",
        "attempts": "Attempts",
        "adversarial": "Adversarial",
        "redacted": "Redacted",
        "errors": "Errors",
        "failure_scenarios": "Failure scenarios",
        "scenarios": "Scenarios",
        "repetitions": "Repetitions",
    }
    columns = [
        "Model",
        "Method",
        "Attempts",
        "Adversarial",
        "Adversarial rate (%)",
        "Adversarial rate SD (pp)",
        "Repetitions",
        "Redacted",
        "Redacted rate (%)",
        "Redacted rate SD (pp)",
        "Errors",
        "Failure scenarios",
        "Scenarios",
    ]
    return summary, table.rename(columns=names)[columns], rates


def format_model_summary(table):
    """Format a copy for display; retain counts, refusal rate and coverage."""
    result = table.copy()
    for metric in ("Adversarial", "Redacted"):
        result[f"{metric} rate (%)"] = [
            f"{mean:.2f} (SD unavailable)" if pd.isna(sd) else f"{mean:.2f} ± {sd:.2f}"
            for mean, sd in zip(
                table[f"{metric} rate (%)"], table[f"{metric} rate SD (pp)"]
            )
        ]
    result["Failure scenarios"] = [
        f"{int(found)}/{int(total)}"
        for found, total in zip(table["Failure scenarios"], table["Scenarios"])
    ]
    return result[
        [
            "Model",
            "Method",
            "Attempts",
            "Adversarial",
            "Adversarial rate (%)",
            "Redacted",
            "Redacted rate (%)",
            "Errors",
            "Failure scenarios",
        ]
    ]


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    # Holm tests the smallest p-value against the largest family multiplier.
    # The running maximum enforces monotonic adjusted p-values in sorted order.
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)

    # Restore the original contrast order before attaching values to tables.
    result[order] = adjusted
    return result


def aligned_scenarios(table, value):
    # Pair by scenario ID, never by incidental input row order. A missing
    # scenario must fail instead of changing the comparison population.
    wide = table.pivot(index="scenario", columns="setting", values=value).sort_index()
    if wide.isna().any().any():
        raise ValueError("All settings must have a nonmissing value for every scenario")
    return wide


def paired_mean_comparisons(table, value, pairs):
    """Paired mean differences, one equally weighted difference per scenario.

    Pairs are (target, reference). This t procedure concerns the mean difference,
    not an odds ratio; the CI is approximate, unadjusted and based on between-
    scenario variation. Holm adjustment covers all supplied pairs as one family.
    It is not repeated-run uncertainty conditional on these fixed 25 scenarios.
    """
    wide = aligned_scenarios(table, value)
    rows = []
    for target, reference in pairs:
        # Each scenario contributes exactly one paired method difference.
        difference = (wide[target] - wide[reference]).to_numpy()
        n = len(difference)
        if n < 2:
            raise ValueError("At least two scenarios are required")
        mean = difference.mean()
        se = difference.std(ddof=1) / np.sqrt(n)
        half_width = stats.t.ppf(0.975, n - 1) * se
        p_value = (
            float(2 * stats.t.sf(abs(mean / se), n - 1))
            if se
            else (1.0 if mean == 0 else 0.0)
        )
        rows.append(
            {
                "target": target,
                "reference": reference,
                "n_scenarios": n,
                "mean_difference": mean,
                "CI_95_low": mean - half_width,
                "CI_95_high": mean + half_width,
                "p_t": p_value,
                "positive_scenarios": int((difference > 0).sum()),
                "negative_scenarios": int((difference < 0).sum()),
                "tied_scenarios": int((difference == 0).sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["p_t_Holm"] = holm_adjust(result.p_t)
    return result


def scenario_sets(frame):
    return {
        setting: set(group.loc[group.adversarial == 1, "scenario"])
        for setting, group in frame.groupby("setting")
    }


def jaccard(a, b):
    # No failures for either method makes failure-set overlap uninformative.
    return len(a & b) / len(a | b) if a | b else np.nan


def observed_overlap(frame):
    sets = scenario_sets(frame)
    return pd.DataFrame(
        [
            {
                "setting_A": a,
                "setting_B": b,
                "Jaccard": jaccard(sets[a], sets[b]),
                "intersection": len(sets[a] & sets[b]),
                "union": len(sets[a] | sets[b]),
                "A_only": sorted(sets[a] - sets[b]),
                "B_only": sorted(sets[b] - sets[a]),
            }
            for a, b in combinations(sets, 2)
        ]
    )


def matched_budget_overlap(frame, target, k=9, draws=20000, seed=20261020):
    """Randomly rarefy baseline to k attempts/scenario, holding target fixed.

    Hypergeometric sampling of success counts is equivalent to uniformly sampling
    baseline rows without replacement for this binary discovery statistic.
    Quantiles describe subset variation conditional on observed outcomes.
    This matches evaluation counts, not compute costs or generation policies.
    """
    table = scenario_table(frame)
    base = table[table.setting == BASELINE].set_index("scenario").sort_index()
    other = table[table.setting == target].set_index("scenario").sort_index()
    if base.empty or other.empty or not base.index.equals(other.index):
        raise ValueError("Baseline and target must cover the same scenarios")
    if not other.attempts.eq(k).all() or base.attempts.lt(k).any():
        raise ValueError(
            f"Target must have {k} attempts/scenario; baseline must have at least {k}"
        )
    if draws < 2:
        raise ValueError("At least two draws required")
    n = base.attempts.to_numpy()
    successes = base.adversarial.to_numpy()

    # Draw success counts rather than physical row subsets. For binary outcomes
    # the hypergeometric law gives the same discovery distribution.
    rng = np.random.default_rng(seed)
    discovered = (
        rng.hypergeometric(successes, n - successes, k, size=(draws, len(n))) > 0
    )
    a = other.adversarial.to_numpy() > 0
    intersections = (discovered & a).sum(axis=1)
    unions = (discovered | a).sum(axis=1)
    similarities = np.divide(
        intersections, unions, out=np.full(draws, np.nan), where=unions != 0
    )

    # Point expectations are exact. Simulation is only needed for conditional
    # subset ranges and nonlinear quantities such as the Jaccard index.
    no_discovery_probability = stats.hypergeom.pmf(0, n, successes, k)
    novel_counts = ((~discovered) & a).sum(axis=1)
    count_a = int(a.sum())
    novelty = novel_counts / count_a if count_a else np.full(draws, np.nan)
    original_base = successes > 0
    return {
        "target": target,
        "attempts_per_scenario": k,
        "target_failure_scenarios": count_a,
        "baseline_full_failure_scenarios": int(original_base.sum()),
        "target_only_vs_full_baseline": int((a & ~original_base).sum()),
        "target_only_scenario_ids": base.index[a & ~original_base].tolist(),
        "target_only_fraction_vs_full_baseline": (
            float((a & ~original_base).sum() / count_a) if count_a else np.nan
        ),
        "expected_baseline_failure_scenarios_at_k": float(
            (1 - no_discovery_probability).sum()
        ),
        "mean_subsampled_Jaccard": float(np.nanmean(similarities)),
        "Jaccard_subset_q025": float(np.nanquantile(similarities, 0.025)),
        "Jaccard_subset_q975": float(np.nanquantile(similarities, 0.975)),
        "expected_target_only_count_at_k": float(no_discovery_probability[a].sum()),
        "expected_target_only_fraction_at_k": (
            float(no_discovery_probability[a].mean()) if count_a else np.nan
        ),
        "target_only_fraction_subset_q025": (
            float(np.nanquantile(novelty, 0.025)) if count_a else np.nan
        ),
        "target_only_fraction_subset_q975": (
            float(np.nanquantile(novelty, 0.975)) if count_a else np.nan
        ),
    }


def within_method_overlap(frame):
    rows = []
    for setting, group in frame.groupby("setting"):
        repetitions = {
            rep: set(g.loc[g.adversarial == 1, "scenario"])
            for rep, g in group.groupby("repetition")
        }
        for a, b in combinations(repetitions, 2):
            rows.append(
                {
                    "setting": setting,
                    "repetition_A": a,
                    "repetition_B": b,
                    "Jaccard": jaccard(repetitions[a], repetitions[b]),
                }
            )
    return pd.DataFrame(rows)


def session_summaries(frame, runs_per_session=3):
    """Describe three-attempt LLM sessions without treating attempts as iid."""
    llm = frame[frame.model != "baseline"]
    # Collapse each feedback chain to one session before describing whether
    # any of its attempts discovered a failure.
    sessions = (
        llm.groupby(["setting", "scenario", "repetition"])
        .agg(
            attempts=("adversarial", "size"),
            failures=("adversarial", "sum"),
        )
        .reset_index()
    )
    if not sessions.attempts.eq(runs_per_session).all():
        raise ValueError("Session metrics require complete sessions")
    sessions["session_success"] = (sessions.failures > 0).astype(int)
    sessions["attempt_yield"] = sessions.failures / sessions.attempts
    # First average repetitions within each scenario, then scenarios.
    summary = (
        sessions.groupby(["setting", "scenario"])[["session_success", "attempt_yield"]]
        .mean()
        .groupby("setting")
        .mean()
    )
    repetition = sessions.groupby(["setting", "repetition"]).agg(
        session_success_rate=("session_success", "mean"),
        mean_attempt_yield=("attempt_yield", "mean"),
        scenarios=("scenario", "nunique"),
    )
    base = scenario_table(frame)
    base = base[base.setting == BASELINE]
    if base.empty or base.attempts.lt(runs_per_session).any():
        raise ValueError("Need enough baseline attempts for a matched session")
    baseline_random_three_success = float(
        (
            1
            - stats.hypergeom.pmf(0, base.attempts, base.adversarial, runs_per_session)
        ).mean()
    )
    return sessions, summary, repetition, baseline_random_three_success


def digit_configuration_comparisons(frame):
    """Exploratory paired-scenario comparisons of both endpoints.

    Holm family: every unordered configuration pair times two endpoints.
    Sign tests are a separately corrected sensitivity family, testing direction
    among non-tied scenarios rather than equality of mean rates. Neither method
    treats the attempts within a scenario as independent replicates.
    """
    digit = frame.loc[frame.model.ne("baseline")]
    table = scenario_table(digit)
    pairs = list(combinations(sorted(digit.setting.unique()), 2))
    if not pairs:
        raise ValueError("At least two DIGIT configurations are required")
    results = []
    for metric in ("adversarial_rate", "redaction_rate"):
        result = paired_mean_comparisons(table, metric, pairs).drop(columns="p_t_Holm")
        wide = aligned_scenarios(table, metric)
        result["metric"] = metric
        sign_p = []
        for target, reference in pairs:
            differences = wide[target] - wide[reference]
            nonzero = differences[differences.ne(0)]
            sign_p.append(
                stats.binomtest(int(nonzero.gt(0).sum()), len(nonzero)).pvalue
                if len(nonzero)
                else 1.0
            )
        result["p_sign"] = sign_p
        result["all_scenarios_tied"] = result.tied_scenarios.eq(result.n_scenarios)
        # A degenerate observed all-zero contrast cannot establish zero risk or
        # population equivalence; do not display a spurious [0, 0] interval.
        result.loc[result.all_scenarios_tied, ["CI_95_low", "CI_95_high"]] = np.nan
        results.append(result)
    combined = pd.concat(results, ignore_index=True)

    # Keep the two scientific family choices explicit. The paper uses the
    # joint family; endpoint-specific columns must not overwrite those values.
    combined["p_t_Holm_all"] = holm_adjust(combined.p_t)
    combined["p_sign_Holm_all"] = holm_adjust(combined.p_sign)
    combined["family_size"] = len(combined)
    combined["p_t_Holm_endpoint"] = combined.groupby("metric")["p_t"].transform(
        holm_adjust
    )
    combined["p_sign_Holm_endpoint"] = combined.groupby("metric")["p_sign"].transform(
        holm_adjust
    )
    combined["endpoint_family_size"] = combined.groupby("metric")["metric"].transform(
        "size"
    )
    return combined
