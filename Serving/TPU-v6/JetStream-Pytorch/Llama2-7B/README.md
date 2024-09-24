# Setup

## Step 0: (optional) Create a virtual environment for Python packages to install

```bash
export WORKDIR=$(pwd)  # set current dir as workdir (can set to something else)
cd $WORKDIR
sudo apt install python3.10-venv
python -m venv venv
source venv/bin/activate
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
pip install -U --pre jax jaxlib libtpu-nightly requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```


Do not install jetstream separately, the above command will install everything.

## Step 2.1: Make sure there is a working version of Jax that can access TPUs:

```bash
python -c "import jax; print(jax.devices())"
```

Should print out something like this:

```bash
(venv) hanq@t1v-n-9c8a4ce2-w-0:/run/user/2003/jetstream-pytorch$ python -c "import jax; print(jax.devices())"
[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=0, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=0, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,3,0), core_on_chip=0)]
```


## Step 3: Get the checkpoint and run conversion

```bash
export input_ckpt_dir=$WORKDIR/ckpt/llama2-7b/original
mkdir -p $input_ckpt_dir

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
python -m convert_checkpoints --model_name=$model_name --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir --quantize_weights=True
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