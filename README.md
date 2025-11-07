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
git clone UNTITLED_REPOSITORY
cd untitled
```

Create a virtual environment and install dependencies.
```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```
