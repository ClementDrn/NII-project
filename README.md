# DIGIT

**Driving Instruction Generation and Injection for Testing** generates passenger
instructions with rule-based or vision-language models and injects them into
SimLingo driving simulations in CARLA.

Version **1.0.0** runs from a source checkout on Linux with Python 3.12 or newer.
Config/jobs files use format **2.0**, and hosts files use **1.0**.

## Setup

```sh
git clone --recurse-submodules https://github.com/ClementDrn/NII-project.git
cd NII-project
uv venv --python 3.12
source .venv/bin/activate
uv sync --frozen --extra dev
cp .env.example .env
cp hosts.example.yaml hosts.yaml
```

Edit `.env` and `hosts.yaml` for your machine and model providers. `HOSTS_PATH`
is relative to the checkout root unless absolute; existing environment variables
override `.env`. Both private files are ignored by Git. Keep the source checkout
and its submodules together when running DIGIT.

The simulation pipeline needs **CARLA 0.9.15** with additional maps, a GPU, and
SimLingo in its own conda environment. Set `CARLA_PATH`, `CONDA_PATH`, and
`OUTPUT_PATH` in `.env`. Follow the pinned
[SimLingo setup instructions](vendor/ads/simlingo/README.md) to install its
environment and model weights.

Remote APIs use the credentials in `.env`. Optional local model instances use
a separate vLLM environment at `vendor/vlm/.venv`; install a vLLM version
compatible with your GPU/CUDA setup and select the instances in `config.yaml`.

## Run

Review `config.yaml` and `jobs.yaml` to select scenarios, generators, filters,
and model instances, then run:

```sh
source .venv/bin/activate
python scripts/run/run_pipeline.py
```

`baseline.yaml` provides a rule-based job configuration. To run it, pass
`--jobs-filepath baseline.yaml`. The first Ctrl+C cancels the pipeline and starts
cleanup; a second interrupts cleanup.

Model requests accept explicit inputs:

```sh
python scripts/vlm/send_remote_llm_request.py \
  --model-instance gpt-5-mini_0 --prompt "Describe a safe following distance."
python scripts/vlm/send_vlm_request.py \
  --model-instance r4b_0 --prompt "Describe these frames." \
  --image /path/to/first.png --image /path/to/second.png
```

Use `--help` on individual scripts for their options. Incremental repairs use
`scripts/run/rerun_incremental_experiments.py` with an explicit experiment
directory and JSON plan such as `{"40": [1, 2]}`. Use `--dry-run` to inspect the
commands first; existing runs are preserved as timestamped backups.

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

Tests use temporary files and scripted model responses; they do not need CARLA,
running model servers, or private configuration. Python callers using configured
pipeline components must call `digit.core.base.initialize_runtime()` explicitly.

Published experiment data are described in [data/DATA.md](data/DATA.md).
Code uses the [MIT License](LICENSE); data use [CC BY 4.0](data/LICENSE).
Vendor components retain their own licenses.
