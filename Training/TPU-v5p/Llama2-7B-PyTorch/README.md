# Instructions for training Llama2-7B with Pytorch/XLA on TPU-V5p

This user guide provides a concise overview of the essential steps required to run StableDiffusion 2.0 base training on Cloud TPUs.


## Environment Setup

The following setup assumes to run the training job with llama2-7b base on GCE TPUs using the docker image from this registery (us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/llama3:v0), the docker image uses the pytorch and torch_xla nightly build from 09/05 and has all the package dependency installed. It cloned the git repo from [https://github.com/pytorch-tpu/diffusers](https://github.com/pytorch-tpu/diffusers/) in order to run hugging face llama2 on TPU. Please follow corresponding TPU generation's user guide to setup the GCE TPUs first. All the command below should run from your own machine (not the TPU host you created).

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
git clone  --depth 1 https://github.com/gclouduniverse/reproducibility.git/
cd reproducibility/Training/TPU-v5p/Llama2-7B-Pytorch-PyTorch
```
2. Edit `env.sh` to add the hugging face token and/or setup the training parameters.
```bash
HF_TOKEN=hf_***
```
3. Run the training script:
```bash
bash benchmark.sh
```
`benchmark.sh` script will upload 1) environment parameters in `env.sh`, 2) model related config in `config.json`, `fsdp_config.json`, 3) docker launch script in `host.sh` and 4) python training command in `train.sh` into all TPU workers, and starts the training afterwards. After the training completes, it will copy the output back under folder *output/*.
