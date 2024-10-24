# Setup
### NOTE 
```
MOE 8x22B optimal implementation and benchmarking is WIP

Currently, it works on v6e-8 and random int8 weights
```


## Step 1: Download JetStream and MaxText github repository
```bash
cd ~
git clone https://github.com/google/maxtext.git
cd maxtext
git checkout rand_int8

cd ~
git clone https://github.com/google/JetStream.git
cd JetStream
git checkout main
```

## Step 2: Setup JetStream and MaxText
```bash
cd ~
sudo apt install python3.10-venv
python -m venv venv-maxtext
source venv-maxtext/bin/activate

cd ~
cd JetStream
pip install -e .
cd benchmarks
pip install -r requirements.in

cd ~
cd maxtext/
bash setup.sh

pip install aqtp==0.7.5

```

# Benchmark

In terminal tab 1, start the server:
```bash
export TOKENIZER_PATH=assets/tokenizer.mistral-v3
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=mixtral-8x22b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=int8
export PER_DEVICE_BATCH_SIZE=36

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  quantize_kvcache=True
```

In terminal tab 2, run the benchmark:
```bash
source venv-maxtext/bin/activate

python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.mistral-v3  \
--warmup-mode sampled   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 1000   \
--max-output-length 1024   \
--dataset openorca
```

(TODO: update) After the benchmark finishes, you should see something like 
```bash

```
