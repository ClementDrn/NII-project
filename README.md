# DIGIT

*Driving Instruction Generation and Injection for Testing* (DIGIT) is a framework that permits the generation of driving instructions by utilizing a vision-language model (VLM) as well as their injection into simulations of the driving agent SimLingo inside CARLA environments.


## Getting Started

Our pipeline requires Linux (tested on Ubuntu 20 & 22) and at least 16 GB of GPU VRAM (for SimLingo and CARLA).

Clone the repository.
```sh
git clone --recursive PATH_TO_REPO
cd digit
```

### Install CARLA and its dependencies

Install CARLA 0.9.15 (36 GB), a version compatible with SimLingo.
```sh
mkdir -p ~/software/carla0915
cd ~/software/carla0915
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
rm CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
```

Install conda if you don't have it already. The lite version called [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#basic-install-instructions) is enough for our purpose.

Copy the [.env.example](.env.example) file as `.env` at repository's root and write down the path to the `CARLA` and `conda` directories at variables `CARLA_PATH` and `CONDA_PATH` respectively.

### Setup Autonomous Driving Agent

Note: Currently, we only support SimLingo ADS.

<!-- TODO: Script for automation??  -->

Set up SimLingo environment.
```sh
# Create base environment
conda env create -f vendor/ads/simlingo/environment.yaml -n simlingo
conda activate simlingo

# Install PyTorch and flash-attn separately to ensure correct CUDA version
pip install torch==2.2.0
pip install flash-attn==2.7.0.post2

# Exit environment not to mix up multiple envs
conda deactivate
```

Download SimLingo model's weights (9.6 GB).
```sh
# Clone by omitting large files (just clone their pointers)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/RenzKa/simlingo vendor/ads/simlingo/outputs/simlingo

# Now, download large files (with progress bar)
cd vendor/ads/simlingo/outputs/simlingo
git lfs pull
cd ../../../../..
```

### Setup VLM server (optional)

You may want not to rely on API model services but instead want to run your own VLM server. 
If so, we show how to do it for the R-4B VLM below (requires more than 8GB of VRAM).

Setup virtual environment with Astral-uv.
```sh
sudo snap install astral-uv --classic
uv venv --python 3.12
source .venv/bin/activate
```

Install `vllm` from source to have the latest version (compatible with R-4B).
```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

Run VLM server with `vllm`.
```sh
vllm serve yannqi/R-4B \
  --served-model-name r4b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

The [Setup Configuration](#setup-configuration) step explains how to specify the server's IP in the project configuration.  

### Install Project Dependencies

Create a virtual environment and install dependencies.
```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .   # if you need dev libs: uv pip install -e ".[dev]" 
```

We just created the project's Python environment containing all necessary dependencies.
It must be activated before running any of this project's scripts. In order to deactivate the environment, just enter command `deactivate`.

Downloading an extra model is required if you wish to use natural language processing features, such as measuring the relevance of generated text. Otherwise, it is not needed.
```sh
python3 -m spacy download en_core_web_trf
```

### Setup Configuration 

If you did not do it yet, copy the [.env.example](.env.example) file as `.env` and change each variable value with your environement settings.

Copy the [hosts.example.yaml](hosts.example.yaml) file as `hosts.yaml`. You may then add IPs and secret API keys for your LLM/VLM services, as well as your own LLM/VLM servers to this file.
If you move this file out of the repository root, then you must write down its new path at variable `HOSTS_PATH` in the `.env` file.

If needed, you can also change the properties in [config.yaml](config.yaml).

### Launch Pipeline

After you have specified what scenarios you want to run in [jobs.yaml](jobs.yaml), you can run the core pipeline with script [scripts/run/run_pipeline.py](scripts/run/run_pipeline.py).
```sh
source .venv/bin/activate
python scripts/run/run_pipeline.py
```


## Tests

Run the suite from the repository root using Python 3.12 or newer:

```sh
source .venv/bin/activate
uv pip install -e ".[dev]"
python -m spacy download en_core_web_trf  # once, for the NLP parsing tests
pytest
```

The suite uses temporary files and scripted model responses. It requires no CARLA
installation, running model servers, API keys, private `.env`/hosts files, or
untracked experiment data. The NLP tests use the locally installed spaCy model;
tests do not download models. Network connections during tests fail explicitly.
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
From a source checkout, install only the analysis dependencies and reproduce the
tables and figures from the experimental data:

```sh
python -m pip install -r scripts/stats/requirements.txt
python scripts/stats/analyze.py reproduce --output-dir output/analysis
```

Open `output/analysis/report.md` for links to the generated results.
Numerical tables, publication figures, and an input/configuration manifest are
saved alongside it. Individual workflows include `effectiveness`,
`complementarity`, `naturalness`, `scene`, `timing`, `diagnostics`, and `validate`.

See [the analysis guide](doc/analysis.md) for formulas, interpretation, input
requirements, installation options, and adapting the study profile. The
[walkthrough notebook](scripts/stats/walkthrough.ipynb) uses the same Python API.


## Provided Data

The data that we collected and used during our experimental analysis are available under the `data/` directory.
You are free to use them under the terms of the [CC-4.0-BY](https://creativecommons.org/licenses/by/4.0/) license.
Check [data/DATA.md](data/DATA.md) for more information.


## License

The code of this project is made available through the MIT License.
Please note that different licenses apply for data files and vendor code.


## Citing

If this project is useful for your work, please cite us.

