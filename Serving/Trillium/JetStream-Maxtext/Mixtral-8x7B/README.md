# Setup
### NOTE 
```
MOE 8x7B optimal implementation and benchmarking is WIP

Currently, it works on v6e-4 with int8 weights
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
export TOKENIZER_PATH=assets/tokenizer.mistral-v1
export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=mixtral-8x7b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=18

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
  compute_axis_order=0,2,1,3 \
  ar_cache_axis_order=0,2,1,3 \
  model_call_mode=inference
```

In terminal tab 2, run the benchmark:
```bash
source venv-maxtext/bin/activate

python JetStream/benchmarks/benchmark_serving.py   \
--tokenizer ~/maxtext/assets/tokenizer.mistral-v1  \
--warmup-mode sampled   \
--save-result   \
--save-request-outputs   \
--request-outputs-file-path outputs.json   \
--num-prompts 2000   \
--max-output-length 1024   \
--dataset openorca
```

After the benchmark finishes, you should see something like 
```bash
Successful requests: 1200
...
Request throughput: 14.93 requests/s
Input token throughput: 3119.41 tokens/s
Output token throughput: 2523.37 tokens/s
Mean TTFT: 160418.68 ms
Median TTFT: 159341.81 ms
P99 TTFT: 316925.23 ms
Mean TPOT: 1457.90 ms
Median TPOT: 846.55 ms
P99 TPOT: 9620.25 ms
....

```
