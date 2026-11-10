# Quantitative analysis

The analysis suite reproduces the quantitative results of *VLM-Guided Test
Generation for Instruction-Following Autonomous Driving Systems* from the CSVs
in `data/`. It also exposes reusable calculations and useful diagnostics for
other experiments. It does not launch simulations, call an LLM, download an
embedding model, or rerun the scene parser.

The reusable statistical code lives in `src/stats`, independently of the
`digit` simulation package. `scripts/stats` contains the command wrapper, paper
profile and walkthrough.

## Install and run

The standalone source-checkout workflow needs Python 3.10+ and only the numerical
dependencies below. Use a virtual environment of your choice:

```sh
python -m pip install -r scripts/stats/requirements.txt
python scripts/stats/analyze.py validate
python scripts/stats/analyze.py reproduce --output-dir output/analysis
```

If Digit is already installed, `python -m pip install -e '.[analysis]'` adds the
same numerical dependencies. This installation method also installs Digit's
existing base dependencies; the standalone requirements avoid that simulation
and model stack. An installed package exposes the equivalent module command:

```sh
python -m stats reproduce --config scripts/stats/paper.yaml
```

The script works from another directory when called by its full path. The default
profile is located relative to the checkout, and its data paths are relative to
the profile rather than your working directory. Output paths are relative to
your working directory. For an installed wheel outside the checkout, supply an
explicit profile path.

```sh
python scripts/stats/analyze.py reproduce \
  --config scripts/stats/paper.yaml --data-dir data --output-dir output/analysis
```

Use `--no-figures` for numerical outputs only. Install the optional notebook
tools with `python -m pip install nbformat nbclient ipykernel` and open
`scripts/stats/walkthrough.ipynb` in your usual Jupyter-capable editor. When using
Digit's extras, the corresponding extra is `.[analysis,notebook]`.

## Choose a workflow

| Command | Calculations | Required data |
| --- | --- | --- |
| `reproduce` | All analysis tables and retained figures | All configured results, annotations, and sampled ratings |
| `validate` | Experimental design, sampled panel, naturalness, and scene checks | Same full dataset as reproduction |
| `effectiveness` | Counts, equal-scenario rates, repetition SDs, paired tests | Result CSVs; baseline and both selected-model variants; at least two generated configurations |
| `complementarity` | Progression, finite-pool novelty, time budgets, overlaps | Results; baseline and both selected-model variants |
| `naturalness` | Sampled estimates/examples and full-pool comparisons | Results, sampled item labels, human/LLM ratings, full-pool scores |
| `scene` | Scene mentions, paired bootstrap, object-type counts | Selected result CSVs with complete count annotations |
| `timing` | Four paper schedules plus baseline, and completion logs | Profile only; result CSVs are not read |
| `diagnostics` | Session success, repetition overlap, full-pool overlap, equal attempts, timing sensitivities | Results |

All commands accept the same path options. Result-based commands validate the
configured result files; missing annotations fail only workflows needing them.
`diagnostics` includes the original 25-worker, joint-repetition, and first-stage
batch scheduling sensitivities, keeping them separate from the paper's four
Table II schedules. Complexity, embedding dispersion, abandoned plots, and old
audit scripts are intentionally outside this suite.

## Analysis outputs

- `tables/`: numerical CSVs, plus readable Markdown versions of Tables I–III.
  Rates, counts, and intervals remain separate numerical columns in CSVs.
- `figures/`: Figures 4a/b, 5a/b, 6a/b, 8, and 9 in PDF/SVG/600-dpi PNG,
  preserving the selected publication sizes, palette, and labels. `captions.md`
  records the plotted quantities and uncertainty definitions.
- `report.md`: links to the generated numerical tables and figures.
- `manifest.json`: input SHA-256 hashes, resolved profile, seeds, dependency and
  Python versions, run time, and output inventory.

Each command computes results from the input data using the study profile.
Missing or malformed required data stops the command with an actionable error.

Use separate output directories when keeping several workflows. Files from an
earlier run are not deleted; the manifest lists only files produced by the
current run, including hashes of the analysis source modules.

Table I includes all six generated configurations. The separate baseline table
retains its two repetitions. Figure 4's numerical export contains the baseline's
entire 658-record tail, even though both plots stop at 225 attempts. Completion
logs and baseline allocations make the Table II and Figure 5 deadlines auditable.
Qualitative Figures 3/7 and individual-record examples are outside the workflow.
Table III remains automated because selecting its examples requires aggregating
880 human ratings and ranking the sampled instructions. Screenshots, method
diagrams, screening logs, and raw timing measurements are not synthesized from
summary CSVs. The configuration documents available design/timing inputs.

## Statistical definitions

The experimental unit for paired method comparisons is the scenario. The
benchmark has 25 systematically selected scenarios. Each generated configuration
has 225 attempts: three runs per scenario in each of three repetitions. Feedback
within an incremental session makes those runs dependent. Baseline has 658
evaluated records across two repetitions and uneven scenario allocation.

**Effectiveness and refusals.** End-to-end success is
`adversarial_raw * usable`, where usable excludes refused/redacted or errored
attempts. Every attempt remains in the denominator. Raw flags and duplicate-text
frequency are retained. Scenario-balanced effectiveness is
`mean_s(successes_s / attempts_s)`. Table I first balances scenarios within each
repetition, then reports the mean and sample SD (`ddof=1`) of repetition rates.
Rate SDs are percentage points, not standard errors or confidence intervals.
Baseline's mean repetition-balanced rate (9.1589%) differs from its rate after
pooling repetitions within scenarios (9.1835%); both round to 9.2%.

Paired two-sided t tests use scenario-rate differences. Paper RQ1 uses Holm
correction jointly across 30 tests (15 pairs × two endpoints), exported as
`p_t_Holm_all`. `p_t_Holm_endpoint` separately corrects 15 comparisons per
endpoint. The corresponding sign-test columns are sensitivity analyses with a
different estimand, not substitutes chosen for significance. Intervals are
pointwise and unadjusted; nonsignificance is not equivalence. The selected
Claude comparison is exploratory selection on the same observations.

**Discovery and novelty.** A failure scenario has at least one observed
end-to-end failure. At sample budget `k`, baseline's no-discovery probability is
`C(N-F,k) / C(N,k)` for each observed scenario pool. Expected target-only count
sums those probabilities over the target's discovered scenarios. Novelty divides
that count by the target's discovery count; an empty target makes it undefined.
These are exact conditional finite-pool expectations. Monte Carlo subset
quantiles describe conditional subset variation, not population CIs.

Progression preserves ascending repetition/scenario/run order and feedback
sessions for DIGIT. Baseline pools repetitions within each scenario and matches
the same scenario order in blocks of three attempts without replacement.
Allocation passes are not baseline experimental repetitions. Partial final
blocks and exhausted pools are retained. Cumulative failure counts, balanced
effectiveness rates, and distinct failure-scenario counts are different outputs.

**Timing.** `G0 → S0 → G1 → S1 → G2 → S2` dependencies are preserved. Default
schedules use one simulator and consecutive repetitions. Mean measured inputs
are 585.94 seconds per simulation and 276.28/839.72/999.88 seconds per generation
stage. Baseline's full 14-second pool-construction cost is paid once for 800
tried candidates; 658 evaluated records are available. The 403-second batch
latency measures first response; delivery of all needed instructions without
subsequent simulator waiting remains an assumption.

Keep equal attempts, the common baseline-completion deadline (36.625139 hours),
and each target's own completion deadline separate. Table II uses the latter;
Figure 5 uses the common deadline and one-worker incremental schedule. Schedules
assume unchanged outcomes under rescheduling and fixed mean service times;
they are modeled estimates, not measured speedups or extrapolations past pools.

**Naturalness.** Explicit sampled-item method labels replace text-membership
inference. There are 40 items per method and 11 complete human raters. Average
human ratings per item before computing variance. Approximate
Welch–Satterthwaite t intervals use `(1-n/N) * sample_variance / n`, with
populations 658 and 225. Paired item-level LLM–human discrepancies retain their
covariance. These intervals condition on the observed panel and judge scores.
Examples select deterministic highest/lowest and lower-middle ranked items,
breaking extrema ties by lowest item ID. The lower-middle example need not
attain the sample median. Human evidence covers incremental only. Full-pool
GPT-5-mini scores cover baseline and both variants and balance scenarios equally.

**Scene relevance.** Average mentions per usable instruction within each
scenario, then average scenarios equally. Metadata-flagged counts are subsets
of totals, and “unflagged” is not independently verified correctness. Flagged
shares divide balanced mean flagged counts by balanced mean totals. Paired
scenario-cluster percentile bootstrap intervals use 20,000 draws, seed
20261022, with the same scenario draws across methods. They are pointwise,
without multiplicity adjustment. Object-type frequencies are mentions per 100
instructions, not proportions of instructions; type counts must sum to
`object_count`, including `unknown`. Cloud position has no semantic meaning.

Scenario-level inference requires defensible independence/exchangeability and
representativeness assumptions. It does not quantify fixed-benchmark uncertainty
from rerunning the entire campaign. Zero observed flags/refusals and zero-width
bootstrap intervals do not establish zero underlying risk.

## Adapt to another study or use Python

Copy `scripts/stats/paper.yaml`, edit the filenames, scenario order, selected
model, repetition/session counts, rating populations, and timing inputs. Paths
are relative to the copied profile. Use the relevant individual command when
your study lacks human ratings or scene annotations. The shared baseline label is
`model=baseline, method=rule-based`; selected-model variants are `one-shot` and
`incremental`. Result IDs start at zero. Timing/progression support the declared
session length; the retained publication annotation positions are tailored to
the paper's three-run, three-repetition design.

```python
from stats.config import load_config
from stats.validation import load_results
from stats.calculations import effectiveness

config = load_config("scripts/stats/paper.yaml", data_dir="data")
frame = load_results(config)
tables = effectiveness(frame, config)  # Pure calculations, no output writes
print(tables["table_I"])
```

`naturalness.analyze_naturalness` returns `(tables, validated_full_pool_records)`;
`scene.analyze_scene` and the functions in `calculations` return numerical
tables. `cli.run_analysis` adds exports, figures, and provenance, returning
`(tables, figure_names)`.

## Verification

The analysis tests are independent of Digit's simulator fixtures. Run them with
the numerical dependencies installed:

```sh
PYTHONPATH=src python -m unittest discover -s tests/analysis -v
```

The tests cover known statistical identities, invalid grids/ratings/joins,
finite-pool enumeration, scheduling dependencies, Holm families, individual
commands, output inventories, and source-file preservation. Execute the
walkthrough from a fresh kernel with notebook tooling installed. After plot
changes, regenerate the paper figures and inspect them at publication size.
