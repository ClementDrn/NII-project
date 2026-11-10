"""Render only the retained paper figures from numerical workflow outputs."""

from pathlib import Path


def create_figures(tables, frame, config, output):
    # Use a noninteractive backend so the same export command works on a
    # workstation or headless server without opening GUI windows.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .config import selected_results
    from .plot_adversarial import plot_failure_overlap, plot_failure_progression
    from .plot_naturalness import plot_naturalness_evidence, save_figure
    from .plot_scene_relevance import plot_object_type_clouds, plot_scene_relevance
    from .publication_style import STYLE

    figures = {}
    if "adversarial_progression" in tables:
        for metric, name in [
            ("cumulative_failures", "figure_4a_failure_outcomes"),
            ("discovered_scenarios", "figure_4b_failure_scenarios"),
        ]:
            figures[name] = plot_failure_progression(
                tables["adversarial_progression"],
                metric,
                config["selected_model"],
                len(config["scenarios"]),
            )[0]
        for method, suffix in [("one-shot", "a"), ("incremental", "b")]:
            figures[f"figure_5{suffix}_overlap"] = plot_failure_overlap(
                selected_results(frame, config),
                tables["novelty_budgets"],
                config["selected_model"],
                panel=method,
            )[0]
    if "naturalness_sample_estimates" in tables:
        full = selected_results(frame, config)
        full.loc[full.model.eq("baseline"), "method"] = "baseline"
        pair, _ = plot_naturalness_evidence(
            tables["naturalness_sample_estimates"], full
        )
        figures.update(
            zip(["figure_6a_sampled_naturalness", "figure_6b_full_naturalness"], pair)
        )
    if "scene_relevance_summary" in tables:
        with plt.rc_context(STYLE):
            figures["figure_8_scene_relevance"] = plot_scene_relevance(
                tables["scene_relevance_summary"], tables["scene_relevance_ci"]
            )[0]
            figures["figure_9_object_types"] = plot_object_type_clouds(
                tables["object_type_frequency"]
            )[0]

    # Export every format from the same figure object to keep annotations,
    # dimensions, and computed values identical across raster/vector outputs.
    directory = Path(output) / "figures"
    try:
        for name, figure in figures.items():
            save_figure(figure, directory / name)
    finally:
        for figure in figures.values():
            plt.close(figure)
    if figures:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "captions.md").write_text(
            "# Figure definitions\n\n"
            "- Figure 4a: cumulative failure outcomes, not scenario-balanced effectiveness percentages. "
            "Figure 4b: distinct scenarios with at least one failure, not distinct failure modes. "
            "DIGIT uses observed repetition/scenario/run replay; baseline uses exact conditional expectations "
            "under uniform within-scenario sampling without replacement, matching three-attempt scenario blocks. "
            "Plots share the DIGIT completion horizon; numerical baseline exports retain the full finite pool.\n"
            "- Figure 5: expected overlap at the common baseline-completion deadline. One-shot is batched; "
            "incremental uses one generation worker and one simulator. Fractional regions are conditional expectations.\n"
            "- Figure 6a: sampled-item means and 95% t intervals with finite-population correction, conditional "
            "on the human panel and observed judge. Human evaluation covers incremental only. "
            "Figure 6b: full-pool scenario-balanced GPT-5-mini means; dots are descriptive scenario means.\n"
            "- Figure 8: equal-scenario mean object/modifier mentions per usable instruction. Hatched overlays "
            "are metadata-flagged subsets, not added counts. Whiskers are pointwise 95% paired scenario-cluster "
            "percentile bootstrap intervals. Percentages divide balanced flagged counts by balanced totals.\n"
            "- Figure 9: scenario-balanced mentions per 100 instructions on one common circle-area scale. "
            "Positions have no semantic meaning; counts are not percentages of instructions.\n",
            encoding="utf-8",
        )
    return list(figures)
