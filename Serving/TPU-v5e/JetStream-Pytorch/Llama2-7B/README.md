# Setup

## Step 0: (optional) Create a virtual environment for Python packages to install

```bash
sudo apt install python3.10-venv
python -m venv venv
source venv/bin/activate
export WORKDIR=$(pwd)  # set current dir as workdir (can set to something else)
```

## Step 1: Get JetStream-PyTorch github repository

```bash
git clone https://github.com/google/jetstream-pytorch.git
cd jetstream-pytorch/
git checkout jetstream-v0.2.3
```

## Step 2: Setup JetStream and JetStream-PyTorch
```bash
source install_everything.sh
```

Do not install jetstream separately, the above command will install everything.

## Step 3: Get the checkpoint and run conversion

```bash
export input_ckpt_dir=$WORKDIR/ckpt/llama2-7b/original

# NOTE: get your own weights from meta!
gcloud storage cp hanq-random/llama-2-7b-chat/* $input_ckpt_dir
```

Run conversion
```bash
export model_name=llama-2
export tokenizer_path=$input_ckpt_dir/tokenizer.llama2

## Step 1: Convert model
export output_ckpt_dir=$WORKDIR/ckpt/llama2-7b/converted
mkdir -p ${output_ckpt_dir}
python -m convert_checkpoints --model_name=$model_name --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir --quantize_weights=True --from_hf=True
```

# Benchmark

In terminal tab 1, start the server:
```bash
export tokenizer_path=$input_ckpt_dir/tokenizer.model
python run_server.py --model_name=$model_name --size=7b --batch_size=96 --max_cache_length=2048 --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml" --quantize_weights=1 --quantize_kv_cache=1

```

In terminal tab 2, run the benchmark:
One time setup
```bash
source venv/bin/activate

export model_name=llama-2
export WORKDIR=$(pwd)  # set current dir as workdir (can set to something else)
export input_ckpt_dir=$WORKDIR/ckpt/llama2-7b/original
export tokenizer_path=$input_ckpt_dir/tokenizer.model

cd jetstream-pytorch/deps/JetStream/benchmarks
pip install -r requirements.in
```

Run the benchmark
```bash
python benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 1000  --dataset openorca --save-request-outputs --warmup-mode=sampled --model=$model_name
```


# NOTE: for release 0.2.4 (coming soon). The commandline interface will change.
See more at https://github.com/google/JetStream-pytorch