# Instructions for training Llama2-7B with Pytorch/XLA on TPU-V5p

This user guide provides a concise overview of the essential steps required to run Llama2-7B training on Cloud TPUs. You can also modify `config.json` to target other llamamodel configurations.


## Environment Setup

The following setup assumes to run the training job with llama2-7b on GCE TPUs using the docker image from this registery (`us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/llama2@sha256:3fda2382a36c8a7c39f8838f9a1abde3a161fd47283b052d04fa090e3ee210f5`), the docker image uses the pytorch and torch_xla nightly build from 09/16/2024 and installed with all the package dependency needed to run the model training. Please follow corresponding TPU generation's user guide to setup the GCE TPUs first. All the command below should run from your own machine (not the TPU host you created).

### Setup Environment of Your TPUs
Please replace all your-* with your TPUs' information.
```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

### Simple Run Command
1. git clone and navigate to this README repo and run training script:
```bash
git clone  --depth 1 https://github.com/gclouduniverse/reproducibility.git
cd reproducibility/Training/TPU-v5p/Llama2-7B-Pytorch
```
2. Edit `env.sh` to add the hugging face token and/or setup the training parameters.
```bash
# add your hugging face token
HF_TOKEN=hf_***
```
3. Edit `host.sh` to add the docker image URL if default docker image is not accessible to you.
```bash
# docker image URL to use for the training
DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/llama2@sha256:3fda2382a36c8a7c39f8838f9a1abde3a161fd47283b052d04fa090e3ee210f5
```
4. Run the training script:
```bash
bash benchmark.sh
```

`benchmark.sh` script will upload 1) environment parameters in `env.sh`, 2) model related configs in `config.json`, `fsdp_config.json`, 3) docker launch script in `host.sh` and 4) python training command in `train.sh` into all TPU workers, and starts the training afterwards. After the training completes, you shall see the train metrics report like below in the terminal for each TPU worker. The script will also copy back the trained model under folder *output/*.
```
[worker :11] [INFO|modelcard.py:450] 2024-09-17 21:45:24,200 >> Dropping the following result as it does not have all the necessary fields:
[worker :11] {'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}, 'dataset': {'name': 'wikitext wikitext-103-raw-v1', 'type': 'wikitext', 'args': 'wikitext-103-raw-v1'}}
[worker :11] {'train_runtime': 808.7671, 'train_samples_per_second': 12.661, 'train_steps_per_second': 0.025, 'train_loss': 9.420475006103516, 'epoch': 0.31}
[worker :11] ***** train metrics *****
[worker :11]   epoch                    =       0.3077
[worker :11]   total_flos               = 1548596160GF
[worker :11]   train_loss               =       9.4205
[worker :11]   train_runtime            =   0:13:28.76
[worker :11]   train_samples            =        33541
[worker :11]   train_samples_per_second =       12.661
[worker :11]   train_steps_per_second   =        0.025
```


### Torch/XLA General Environment Envs Explained

*   `DOCKER_IMAGE`: Docker registry URL of the image to pull on all TPU workers.
*   `PJRT_DEVICE`: Specify the XLA device.
*   `XLA_USE_SPMD`: Turn on GSPMD.
*   `XLA_IR_DEBUG`: Capture Python stack trace in Lazy IRs.
*   `XLA_HLO_DEBUG`: Capture Python stack trace in HLOs.
*   `PROFILE_EPOCH`: Specify which epoch to start taking the profile.
*   `PROFILE_STEP`: Specify which step to start taking the profile.
*   `PROFILE_DURATION_MS`: Specify how long the profiling will last.


### HF Llama Arguments Explained

*   `--flash_attention`: [bool] Enable Pallas FlashAttention. Default: False.
*   `--per_device_train_batch_size`: [int] Specify the global batch size. GSPMD treats the program as a singel device program.
