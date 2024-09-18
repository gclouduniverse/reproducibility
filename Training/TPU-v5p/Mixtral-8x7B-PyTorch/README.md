# Instructions for training  Mixtral-8X7B on TPU v5p



This user guide provides a concise overview of the essential steps required to run HuggingFace (HF) Mixtral training on Cloud TPUs.


## Environment Setup

Please follow the corresponding TPU generation's user guide to setup the GCE TPUs first.
### Setup Environment of Your TPUs
Please replace all your-* with your TPUs' information.
```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

### HF Mixtral 7 x 8B Environment Setup (without docker)

<details>

<summary>Click to see how to setup environment and run without docker</summary>

The following setup assumes to run the training job with Mixtral 7 x 8B on GCE TPUs. Please follow corresponding TPU generation's user guide to setup the GCE TPUs. For GKE users, most of the commands below also apply.

Here both PyTorch and PyTorch/XLA nightly are used with our fork of HuggingFace.
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Step 1: install torch, torch-xla, libtpu, pallas
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl
pip3 install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
pip3 install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Step 2: install HF
git clone -b alanwaketan/moe https://github.com/pytorch-tpu/transformers.git
cd transformers
pip3 install git+file://$PWD
pip3 install accelerate datasets evaluate scikit-learn huggingface-hub
'
```

The next step is to sign into HF such that you can get accesses to the tokenizer or model checkpoints. Please replace `your_token` with your HF token.
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
export PATH=$PATH:/home/$USER/.local/bin
huggingface-cli login --token your_token
'
```

The next step for HF setup is to copy your [Mixtral config](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) into the TPU VM.
```
gcloud compute tpus tpu-vm scp mixtral87.json $TPU_NAME:~/config.json --worker all --project $PROJECT --zone=$ZONE
```

The last step for HF setup is to copy your fsdp_config.json into the TPU VM.
```
{
    "fsdp_transformer_layer_cls_to_wrap": [
        "MixtralDecoderLayer"
    ],
    "xla": true,
    "xla_fsdp_v2": true,
    "xla_fsdp_grad_ckpt": true
}

```
And the command to copy the config.
```
gcloud compute tpus tpu-vm scp fsdp_config.json $TPU_NAME:~/fsdp_config.json --worker all --project $PROJECT --zone=$ZONE
```

## Steps to Run HF Mixtral 8 x 7B
Following is the gcloud ssh command to run the training job from the host:
```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone ${ZONE} \
--project ${PROJECT} \
--worker=all \
--command='
# Setup envs
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/

# Run
cd transformers
python3 examples/pytorch/language-modeling/run_clm.py \
  --dataset_name wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --per_device_train_batch_size 32 \
  --do_train \
  --output_dir /tmp/test-clm \
  --overwrite_output_dir \
  --config_name ~/config.json \
  --cache_dir /tmp \
  --tokenizer_name mistralai/Mixtral-8x7B-v0.1 \
  --block_size 4096 \
  --optim adafactor \
  --save_strategy no \
  --logging_strategy no \
  --fsdp "full_shard" \
  --fsdp_config ~/fsdp_config.json \
  --torch_dtype bfloat16 \
  --dataloader_drop_last yes \
  --flash_attention \
  --max_steps 10 \
  --gmm

```
</details>  

------  
### HF Mixtral 7 x 8B Environment Setup (with docker)

The following setup assumes to run the training job with Mixtral 7 x 8B on GCE TPUs using the docker image from this registery (`us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral@sha256:1e9024d13e53bdadc13d7a695d8fbe52b95a78cfae2a101da8a9d8fc94b1c96b`), the docker image uses the pytorch and torch_xla nightly build from 09/16/2024 and installed with all the package dependency needed to run the model training. All the command below should run from your own machine (not the TPU host you created).

1. git clone and navigate to this README repo and run training script:
```bash
git clone  --depth 1 https://github.com/gclouduniverse/reproducibility.git
cd reproducibility/Training/TPU-v5p/Mixtral-8x7B-PyTorch
```
2. Edit `env.sh` to add the hugging face token and/or setup the training parameters.
```bash
# add your hugging face token
HF_TOKEN=hf_***
```
3. Edit `host.sh` to add the docker image URL if default docker image is not accessible to you.
```bash
# docker image URL to use for the training
DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral@sha256:1e9024d13e53bdadc13d7a695d8fbe52b95a78cfae2a101da8a9d8fc94b1c96b
```
4. Run the training script:
```bash
bash benchmark.sh
```
`benchmark.sh` script will upload 1) environment parameters in `env.sh`, 2) model related config in `config.json`, `fsdp_config.json`, 3) docker launch script in `host.sh` and 4) python training command in `train.sh` into all TPU workers, and starts the training afterwards. When all training steps complete, it will print out training metrics of each worker as below in terminal:
```
[worker :0] ***** train metrics *****
[worker :0]   epoch                    =        0.3125
[worker :0]   total_flos               = 10915247040GF
[worker :0]   train_loss               =         9.278
[worker:0]   train_runtime            =    0:46:45.60
[worker :0]   train_samples            =         32816
[worker :0]   train_samples_per_second =          3.65
[worker :0]   train_steps_per_second   =         0.007
```
In addition,  it will copy back the trained model under *output/*.
