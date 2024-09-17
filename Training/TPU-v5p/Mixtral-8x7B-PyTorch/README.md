# Instructions for training  Mixtral-8X7B on TPU v5p


This user guide provides a concise overview of the essential steps required to run  Mixtral-8X7B training on Cloud TPUs.


## Environment Setup

The following setup assumes to run the training job with Mixtral-8X7B base on GCE TPUs using the docker image from this registery (us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral), the docker image uses the pytorch and torch_xla  with all the package dependency installed. Please follow corresponding TPU generation's user guide to setup the GCE TPUs first. All the command below should run from your own machine (not the TPU host you created).

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
`benchmark.sh` script will upload 1) environment parameters in `env.sh`, 2) model related config in `config.json`, `fsdp_config.json`, 3) docker launch script in `host.sh` and 4) python training command in `train.sh` into all TPU workers, and starts the training afterwards. When all training steps complete, it will print out the training metrics or each worker as below in terminal:
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
In addition,  it will copy back the trained model output under *output/*.
