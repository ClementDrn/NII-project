# Adversarial Instruction Testing for VLA4AD

## Running Prototype

### Setup VLM server

Setup virtual environment with Astral-uv and install `vllm`.
```sh
sudo snap install astral-uv --classic
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm
```

Run VLM server with `vllm`.
```sh
vllm serve yannqi/R-4B \
  --served-model-name r4b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --swap-space 16 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

### Setup App

Clone the repository.
```sh
git clone --recursive UNTITLED_REPOSITORY
cd untitled
```

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

Download SimLingo model's weights (12GB).
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
uv pip install -e .
```

### Setup Environement Configuration 

In the `.env` file, change each variable value with your environement settings.
