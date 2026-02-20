# Adversarial Instruction Testing for VLA4AD

## Getting Started

Our pipeline requires Linux (tested on Ubuntu 20 & 22) and at least 16 GB of GPU VRAM (for SimLingo and CARLA).

Clone the repository.
```sh
git clone --recursive UNTITLED_REPOSITORY
cd untitled
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

Create a virtual environment and install dependencies.
```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .   # if you need dev libs: uv pip install -e ".[dev]" 
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

The next step explains how to specify the server's IP in the project configuration.  

### Setup Configuration 

If you did not do it yet, copy the [.env.example](.env.example) file as `.env` and change each variable value with your environement settings.

Copy the [hosts.example.yaml](hosts.example.yaml) file as `hosts.yaml`. You may then add IPs and secret API keys for your LLM/VLM services, as well as your own LLM/VLM servers to this file.
If you move this file out of the repository root, then you must write down its new path at variable `HOSTS_PATH` in the `.env` file.

If needed, you can also change the properties in [config.yaml](config.yaml).

### Launch Prototype

After you have specified what scenarios you want to run in [jobs.yaml](jobs.yaml), you can run the [prototype.py](scripts/prototype.py) script.
```
source .venv/bin/activate
python scripts/prototype.py
```
