# Setup
### NOTE 
```
MOE 8x22B optimal implementation and benchmarking is WIP

Currently, it works on v6e-8 and int8 weights
```


## Step 1: Download JetStream and MaxText github repository
```bash
cd ~
git clone https://github.com/google/maxtext.git
cd maxtext


cd ~
git clone https://github.com/google/JetStream.git
cd JetStream
git checkout v0.3
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
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=24

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
  quantization=int8 \
  quantize_kvcache=True \
  checkpoint_is_quantized=True \
  attention=dot_product \
  megablox=False \
  sparse_matmul=False \
  model_call_mode=inference
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

After the benchmark finishes, you should see something like 
```bash
Successful requests: 995
Benchmark duration: 445.972525 s
Total input tokens: 210595
Total generated tokens: 1020870
Request throughput: 2.23 requests/s
Input token throughput: 472.22 tokens/s
Output token throughput: 2289.09 tokens/s
Mean TTFT: 191650.58 ms
Median TTFT: 186643.05 ms
P99 TTFT: 389921.50 ms
Mean TPOT: 250.54 ms
Median TPOT: 244.73 ms
P99 TPOT: 434.34 ms
...
```
