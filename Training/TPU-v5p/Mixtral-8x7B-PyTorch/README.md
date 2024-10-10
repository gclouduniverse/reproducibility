# Instructions for training Mixtral-8X7B on TPU v5p


This user guide provides a concise overview of the essential steps required to run HuggingFace (HF) Mixtral training on Cloud TPUs.


## Environment Setup

Please follow the corresponding TPU generation's user guide to setup the GCE TPUs
first.

NOTE: For best performance on Mixtral, use "untwisted" variants of TPU topologies.
For example, if you're creating a 128 chip v5p slice, select `4x4x8_untwisted` in
the `gcloud` CLI. The default is "twisted" which has been observed to reduce
performance in Mixtral. See [1] about details on twisted tori.

Please replace all your-* with your TPUs' information.

```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

You may use this command to create an untwisted 128 chip v5p slice:

```
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v5p --topology 4x4x8_untwisted \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv5
```

## Steps to Run HF Mixtral 8x7B

The following setup runs the training job with Mixtral 8x7B on GCE TPUs using the docker image from this registry (`us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral@sha256:c8f4a66e02a26548c9d71296cd345d3a302f6960db3da7fd3addf34c00332b5b`), the docker image uses the pytorch and torch_xla nightly build from 09/28/2024 and installed with all the package dependency needed to run the model training. All the command below should run from your own machine (not the TPU host you created).

1. git clone and navigate to this README repo and run training script:
```bash
git clone --depth 1 https://github.com/gclouduniverse/reproducibility.git
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
DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral@sha256:c8f4a66e02a26548c9d71296cd345d3a302f6960db3da7fd3addf34c00332b5b
```
4. Run the training script:
```bash
./benchmark.sh
```
`benchmark.sh` script will upload 1) environment parameters in `env.sh`, 2) model related config in `config.json`, `fsdp_config.json`, 3) docker launch script in `host.sh` and 4) python training command in `train.sh` into all TPU workers, and starts the training afterwards. When all training steps complete, it will print out training metrics of each worker as below in terminal:
```
[worker :0] ***** train metrics *****
[worker :0]   epoch                    =        0.3125
[worker :0]   total_flos               = 10915247040GF
[worker :0]   train_loss               =         9.278
[worker :0]   train_runtime            =    0:46:45.60
[worker :0]   train_samples            =         32816
[worker :0]   train_samples_per_second =          3.65
[worker :0]   train_steps_per_second   =         0.007
```
In addition,  it will copy back the trained model under `output/*`.

<!-- xrefs -->

[1]: https://cloud.google.com/tpu/docs/v4#twisted-tori

