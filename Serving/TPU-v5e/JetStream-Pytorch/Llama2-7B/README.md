# Setup

## Step 1: Download JetStream and JetStream-PyTorch github repository
```bash
cd ~
git clone https://github.com/google/jetstream-pytorch.git
cd jetstream-pytorch/
git checkout jetstream-v0.2.3

cd ~
git clone https://github.com/google/JetStream.git
cd JetStream
git checkout main
git pull origin main
```

## Step 2: Setup JetStream and JetStream-PyTorch
```bash
cd ~
python -m venv venv-pt
source venv-pt/bin/activate

cd ~/JetStream
pip install -e .
cd benchmarks
pip install -r requirements.in

cd ~/jetstream-pytorch
source install_everything.sh
```

## Step 3: Get the checkpoint and run conversion
# The commands may be required for gsutil
# sudo apt-get install gcc python3-dev python3-setuptools
# sudo pip3 uninstall crcmod
# sudo pip3 install --no-cache-dir -U crcmod

```bash
export META_CHECKPOINT_PATH=gs://maxtext-llama/llama2-7b/meta-ckpt
export input_ckpt_dir=~/ckpt/llama2-7b/original
mkdir -p ${input_ckpt_dir}
gsutil -m cp -r ${META_CHECKPOINT_PATH}/* ${input_ckpt_dir}

export TOKENIZER_PATH=gs://maxtext-llama/llama2-7b/tokenizer.llama2
gsutil cp ${TOKENIZER_PATH} ~
```

Run conversion
```bash
export model_name=llama-2
export tokenizer_path=~/tokenizer.llama2

## Step 1: Convert model
export output_ckpt_dir=~/ckpt/llama2-7b/converted
mkdir -p ${output_ckpt_dir}
python -m convert_checkpoints --model_name=$model_name --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir
```

# Benchmark

In terminal tab 1, start the server:
```bash
cd ~/jetstream-pytorch
python run_server.py --model_name=$model_name --size=7b --batch_size=24 --max_cache_length=2048 --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml"

```

In terminal tab 2, run the benchmark:
```bash
source venv-pt/bin/activate

export model_name=llama-2
export tokenizer_path=~/tokenizer.llama2

cd ~/JetStream
python benchmarks/benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 1000  --dataset openorca --save-request-outputs --warmup-mode=sampled --model=$model_name

```
