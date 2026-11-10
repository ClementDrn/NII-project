# DIGIT

**Driving Instruction Generation and Injection for Testing** generates passenger instructions with rule-based or vision-language models and injects them into SimLingo driving simulations in CARLA.

## Setup

```sh
git clone --recurse-submodules https://github.com/ClementDrn/digit.git
cd digit
uv venv --python 3.12
source .venv/bin/activate
uv sync --frozen --extra dev
cp .env.example .env
cp hosts.example.yaml hosts.yaml
```

Edit `.env` and `hosts.yaml` for your machine and model providers. `HOSTS_PATH` is relative to the checkout root unless absolute; existing environment variables override `.env`. Both private files are ignored by Git.
Keep the source checkout and its submodules together when running DIGIT.

The simulation pipeline needs **CARLA 0.9.15** with additional maps, a GPU, and SimLingo in its own conda environment. Set `CARLA_PATH`, `CONDA_PATH`, and `OUTPUT_PATH` in `.env`.
Follow the pinned [SimLingo setup instructions](vendor/ads/simlingo/README.md) to install its environment and model weights.

Remote APIs use the credentials in `.env`. Optional local model instances use a separate vLLM environment at `vendor/vlm/.venv`; install a vLLM version compatible with your GPU/CUDA setup and select the instances in `config.yaml`.

## Run

Review `config.yaml` and `jobs.yaml` to select scenarios, generators, filters,
and model instances, then run:

```sh
source .venv/bin/activate
python scripts/run/run_pipeline.py
```

`baseline.yaml` provides a rule-based job configuration. To run it, pass `--jobs-filepath baseline.yaml`.

The first Ctrl+C cancels the pipeline and starts cleanup; a second interrupts cleanup.

Model requests accept explicit inputs:

```sh
python scripts/vlm/send_remote_llm_request.py \
  --model-instance gpt-5-mini_0 --prompt "Describe a safe following distance."
python scripts/vlm/send_vlm_request.py \
  --model-instance r4b_0 --prompt "Describe these frames." \
  --image /path/to/first.png --image /path/to/second.png
```

Use `--help` on individual scripts for their options.
Incremental repairs use `scripts/run/rerun_incremental_experiments.py` with an explicit experiment directory and JSON plan such as `{"40": [1, 2]}`.
Use `--dry-run` to inspect the commands first; existing runs are preserved as timestamped backups.

## Results and tests

```sh
python scripts/results/save_experiment_data.py path/containing/jobs results.csv
python scripts/evaluate/analyze_scene_objects.py results_job_000.csv objects.csv
python scripts/evaluate/evaluate_naturalness_of_instructions.py objects.csv scored.csv
```

CSV export appends `_job_<id>` to the requested filename. Pass `-H` to enable
hallucination checks. NLP analysis and its integration tests need the spaCy model:

```sh
source .venv/bin/activate
uv pip install pip
python -m spacy download en_core_web_trf
pytest
```

The suite uses temporary files and scripted model responses.
It requires no CARLA installation, running model servers, API keys, private `.env`/hosts files, or untracked experiment data.
The NLP tests use the locally installed spaCy model; tests do not download models. Network connections during tests fail explicitly.
Configuration integration tests validate the committed files and example settings.

## Scripts

Run scripts from the repository root with the project's virtual environment activated. The `scripts/` directories contain:

- `run/`: launch pipelines, simulations, and experiment reruns.
- `vlm/`: generate instructions and try model requests, including API batches.
- `test/`: inspect frame filtering and instruction rules.
- `results/`: display, summarize, and export experiment results.
- `evaluate/`: evaluate instruction naturalness and count scene object mentions.
- `utils/`: extract instructions, export failed runs, and merge job directories.
- `stats/`: perform statistical analysis over the results.

A typical workflow is:
```sh
# Run experiments from jobs.yaml
python scripts/run/run_pipeline.py

# Export results to CSV (and check for instruction hallucinations)
python scripts/results/save_experiment_data.py -H path/containing/jobs results.csv

# Export the different scene object categories mentioned
python scripts/evaluate/analyze_scene_objects.py results.csv results-o.csv
# Evaluate naturalness of instructions with a GPT-5-mini
python scripts/evaluate/evaluate_naturalness_of_instructions.py results-o.csv results-x.csv

# Reproduce the paper's quantitative analysis from the provided CSVs
python scripts/stats/analyze.py reproduce --output-dir output/analysis
```

Use `--help` on these scripts for their options.

### Quantitative analysis

The statistical code lives in `src/stats`, separate from the `digit` package.
Analysis can run independently of CARLA, SimLingo, GPU models, and API services.
From a source checkout, install only the analysis dependencies and reproduce the tables and figures from the experimental data:
```sh
python -m pip install -r scripts/stats/requirements.txt
python scripts/stats/analyze.py reproduce --output-dir output/analysis
```

Open `output/analysis/report.md` for links to the generated results.
Numerical tables, publication figures, and an input/configuration manifest are saved alongside it.
Individual workflows include `effectiveness`, `complementarity`, `naturalness`, `scene`, `timing`, `diagnostics`, and `validate`.

See [the analysis guide](doc/analysis.md) for formulas, interpretation, input requirements, installation options, and adapting the study profile.
The [walkthrough notebook](scripts/stats/walkthrough.ipynb) uses the same Python API.

## Provided Data

The data that we collected and used during our experimental analysis are available under the `data/` directory.
You are free to use them under the terms of the [CC-4.0-BY](https://creativecommons.org/licenses/by/4.0/) license.
Check [data/DATA.md](data/DATA.md) for more information.

## License

The code of this project is made available through the [MIT License](LICENSE).
Please note that different licenses apply for data files and vendor code.

## Citing

If this project is useful for your work, please cite us.
