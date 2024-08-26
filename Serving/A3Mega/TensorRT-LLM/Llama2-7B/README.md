Install TensorRT-LLM
```
sudo apt-get update
sudo apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout v0.11.0
git submodule update --init --recursive
git lfs install
git lfs pull

sudo nvidia-smi -pm 1

make -C docker release_build
make -C docker release_run DOCKER_RUN_ARGS="-v /scratch:/scratch"
```
Python benchmark

float16
```
cd /app/tensorrt_llm/examples/llama
python convert_checkpoint.py --model_dir /scratch/models/Llama-2-7b-chat-hf                             --output_dir /scratch/models/tllm_checkpoint_1gpu_fp16                             --dtype float16 --tp_size 1
trtllm-build --checkpoint_dir /scratch/models/tllm_checkpoint_1gpu_fp16/              --output_dir  /scratch/engines/llama/7B-chat/trt_engines/fp16/1-gpu             --gemm_plugin auto
```

fp8
```
cd /app/tensorrt_llm/examples/llama
python examples/quantization/quantize.py --dtype=float16  --output_dir=/scratch/models/fp8-quantized-ammo/llama2-7b-tp2pp1-fp8 --model_dir=/scratch/models/Llama-2-7b-chat-hf --qformat=fp8 --kv_cache_dtype=fp8 --tp_size 1
trtllm-build --checkpoint_dir /scratch/models/fp8-quantized-ammo/llama2-7b-chat-hf-tp1pp1-fp8/              --output_dir  /scratch/engines/llama/7B-chat/trt_engines/fp8/1-gpu             --gemm_plugin auto
```

```
cd benchmarks/python
python benchmark.py -m llama_7b --mode plugin --batch_size "64"     --input_output_len "2048,2048"
[TensorRT-LLM] TensorRT-LLM version: 0.11.0
Allocated 1112.00 MiB for execution context memory.
/usr/local/lib/python3.10/dist-packages/torch/nested/__init__.py:219: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
  return _nested.nested_tensor(
[08/21/2024-23:21:05] [TRT-LLM] [W] Logger level already set from environment. Discard new verbosity: error
[TensorRT-LLM] TensorRT-LLM version: 0.11.0
[BENCHMARK] model_name llama_7b world_size 1 num_heads 32 num_kv_heads 32 num_layers 32 hidden_size 4096 vocab_size 32000 precision float16 batch_size 64 gpu_weights_percent 1.0 input_length 128 output_length 128 gpu_peak_mem(gb) 20.546 build_time(s) 0 tokens_per_sec 7373.14 percentile95(ms) 1116.268 percentile99(ms) 1118.393 latency(ms) 1111.06 compute_cap sm90 quantization QuantMode.FP8_QDQ|FP8_KV_CACHE generation_time(ms) 982.682 total_generated_tokens 8128.0 generation_tokens_per_second 8271.238
```



CPP benchmark

Download checkpoint from gs://vertex-model-garden-public-us-central1/llama2

Build wheel
```
python3 ./scripts/build_wheel.py  --trt_root /usr/local/tensorrt
```

Build TenorRT-LLM engine

Build benchmark
```
cd cpp/build
make -j benchmarks
```

Run the benchmark
```
./benchmarks/gptManagerBenchmark     --engine_dir /scratch/engines/llama/7B-chat/trt_engines/fp8/1-gpu/     --request_rate 10     --dataset /scratch/data/sharegpt_llama3_300.json     --max_num_samples 500
[BENCHMARK] num_samples 500
[BENCHMARK] num_error_samples 0

[BENCHMARK] num_samples 500
[BENCHMARK] total_latency(ms) 52372.21
[BENCHMARK] seq_throughput(seq/sec) 9.55
[BENCHMARK] token_throughput(token/sec) 2870.80

[BENCHMARK] avg_sequence_latency(ms) 2755.07
[BENCHMARK] max_sequence_latency(ms) 3198.22
[BENCHMARK] min_sequence_latency(ms) 2353.59
[BENCHMARK] p99_sequence_latency(ms) 3108.04
[BENCHMARK] p90_sequence_latency(ms) 2942.41
[BENCHMARK] p50_sequence_latency(ms) 2747.11
```
Run with Triton + TensorRT-LLM
Download triton inference server
```
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
# Update the submodules
cd tensorrtllm_backend
git checkout r24.04
git submodule update --init --recursive
git lfs install
git lfs pull
```
Launch docker 
```
sudo docker run --rm -it --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -p 7080:7080 -v /scratch/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3 bash
```
Build TensorRT-LLM
```
cd /tensorrtllm_backend/tensorrt_llm
python3 ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt"
pip3 install ./build/tensorrt_llm*.whl
```
Build TenorRT-LLM engine
```
export HF_LLAMA_MODEL=/scratch/models/Llama-2-7b-chat-hf/
export UNIFIED_CKPT_PATH=/scratch/models/tllm_checkpoint_1gpu_fp16/
export ENGINE_PATH=/scratch/engines/llama/7B-chat/triton/fp16/1-gpu/

python convert_checkpoint.py --model_dir ${HF_LLAMA_MODEL} \
                             --output_dir ${UNIFIED_CKPT_PATH} \
                             --dtype float16

trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${ENGINE_PATH} \
             --paged_kv_cache enable \
             --max_batch_size 64
```
Prepare configs
```
cp all_models/inflight_batcher_llm/ llama_ifb -r

python3 tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
python3 tools/fill_template.py -i llama_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
python3 tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64
python3 tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
```
Launch Triton server
```
pip install tritonclient[all]
CUDA_VISIBILE_DEVICES=0 python3 scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/
```
Send request
```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}'
```
Send request with bad_words and stop_words
```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": ["intelligence", "model"], "stop_words": ["focuses", "learn"], "pad_id": 2, "end_id": 2}'
```
Run the benchmark
```
python3 tools/inflight_batcher_llm/end_to_end_test.py --dataset ci/L0_backend_trtllm/simple_data.json --max-input-len 500
/usr/local/lib/python3.10/dist-packages/tritonclient/grpc/service_pb2_grpc.py:21: RuntimeWarning: The grpc package installed is at version 1.60.1, but the generated code in grpc_service_pb2_grpc.py depends on grpcio>=1.64.1. Please upgrade your grpc module to grpcio>=1.64.1 or downgrade your generated code using grpcio-tools<=1.60.1. This warning will become an error in 1.65.0, scheduled for release on June 25, 2024.
  warnings.warn(
[INFO] Start testing on 13 prompts.
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
context_logits.shape: (1, 1, 1)
generation_logits.shape: (1, 1, 1, 1)
[INFO] Functionality test succeed.
[INFO] Warm up for benchmarking.
[INFO] Start benchmarking on 13 prompts.
[INFO] Total Latency: 349.046 ms
```
