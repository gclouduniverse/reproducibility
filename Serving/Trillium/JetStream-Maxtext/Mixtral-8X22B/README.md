**Please run setup and benchmark steps on v6e-8 unless otherwise specified.**

# Setup

## Step 1: Download JetStream and MaxText github repository
```bash
cd ~
git clone https://github.com/google/maxtext.git

cd ~
git clone https://github.com/google/JetStream.git
```

## Step 2: Set up JetStream and MaxText
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
bash setup.sh MODE=stable DEVICE=tpu
pip install nltk==3.8.1
```

## (Optional) Step 3: Checkpoint conversion
If you don't have a MaxText compatible checkpoint, follow these steps [here](#Appendix-checkpoint-conversion-on-CPU).

# Benchmark

In terminal tab 1, start the server:
```bash
export LOAD_PARAMETERS_PATH=gs://...
export TOKENIZER_PATH=assets/tokenizer.mistral-v3
export MAX_PREFILL_PREDICT_LENGTH=1024
export MAX_TARGET_LENGTH=2048
export MODEL_NAME=mixtral-8x22b
export ICI_FSDP_PARALLELISM=1
export ICI_AUTOREGRESSIVE_PARALLELISM=1
export ICI_TENSOR_PARALLELISM=-1
export SCAN_LAYERS=false
export WEIGHT_DTYPE=bfloat16
export PER_DEVICE_BATCH_SIZE=8

cd ~/maxtext
python MaxText/maxengine_server.py \
  MaxText/configs/base.yml \
  tokenizer_path=${TOKENIZER_PATH} \
  load_parameters_path=${LOAD_PARAMETERS_PATH} \
  max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
  max_target_length=${MAX_TARGET_LENGTH} \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
  ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
  ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
  scan_layers=${SCAN_LAYERS} \
  weight_dtype=${WEIGHT_DTYPE} \
  per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
  megablox=False \
  capacity_factor=-1 \
  quantization=int8 checkpoint_is_quantized=True \
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
--num-prompts 1200   \
--max-output-length 1024   \
--dataset openorca  \
--run-eval true
```

After the benchmark finishes, you should see something like 
```bash
Successful requests: 1200
Benchmark duration: 1325.742353 s
Total input tokens: 254423
Total generated tokens: 1212869
Request throughput: 0.91 requests/s
Input token throughput: 191.91 tokens/s
Output token throughput: 914.86 tokens/s
Mean TTFT: 841315.81 ms
Median TTFT: 861898.76 ms
P99 TTFT: 1283356.98 ms
Mean TPOT: 1222.36 ms
Median TPOT: 884.48 ms
P99 TPOT: 16942.69 ms
```

# Appendix Checkpoint conversion on CPU

```bash
# Get checkpoint from https://github.com/mistralai/mistral-inference
# The ckeckpoint size is around 260 GB
export M8x22B_DIR=/tmp/8x22b_instruct
wget https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-Instruct-v0.3.tar
mkdir -p ${M8x22B_DIR}
tar -xf mixtral-8x22B-Instruct-v0.3.tar -C ${M8x22B_DIR}

# Convert checkpoint to scanned version
# The current script llama_or_mistral_ckpt.py only support conversion from `.pth` format
export CHKPT_BUCKET=gs://<your_bucket_name>
export SCANNED_CHKPT_PATH=${CHKPT_BUCKET}/scanned_ckpt
JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py \
--base-model-path=${M8x22B_DIR} --model-size=mixtral-8x22b \
--maxtext-model-path=${SCANNED_CHKPT_PATH}

# Convert checkpoint to unscanned version
export UNSCANNED_RUN_NAME=unscanned_ckpt
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py \
MaxText/configs/base.yml async_checkpointing=false \
base_output_directory=${CHKPT_BUCKET} load_parameters_path=${SCANNED_CHKPT_PATH}/0/items \
run_name=${UNSCANNED_RUN_NAME} model_name='mixtral-8x22b' force_unroll=true

# Convert checkpoint to quantized version
export QUANTIZED_CHKPT_PATH=${CHKPT_BUCKET}/quantized_ckpt
JAX_PLATFORMS=cpu python MaxText/decode.py MaxText/configs/base.yml \
hardware=cpu tokenizer_path=assets/tokenizer.mistral-v3 
load_parameters_path=${CHKPT_BUCKET}/${UNSCANNED_RUN_NAME}/0/items \
model_name=mixtral-8x22b ici_fsdp_parallelism=1 ici_tensor_parallelism=1 \
scan_layers=false weight_dtype=bfloat16 per_device_batch_size=1 \
attention=dot_product megablox=False capacity_factor=-1 quantization=int8 \
quantize_kvcache=True save_quantized_params_path=${QUANTIZED_CHKPT_PATH}
```