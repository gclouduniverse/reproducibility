# User Guide: Running HuggingFace Mixtral Training on Cloud TPUs


This user guide provides a concise overview of the essential steps required to run StableDiffusion 2.1 training on Cloud TPUs.


## Environment Setup

The following setup assumes to run the training job with StableDiffusion 2.1 on GCE TPUs using the docker image from [this registery]( us-central1-docker.pkg.dev/tpu-pytorch/docker/development/pytorch-tpu-diffusers:v1), which has all the package dependency installed. Please follow corresponding TPU generation's user guide to setup the GCE TPUs.

### Setup Environment of Your TPUs
Please replace all your-* with your TPUs' information.
```
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
```

### Simple Run Command
Navigate to this README repo and run the following command:
```bash
bash benchmark.sh
```
The shell script will copy 1) environment parameters in `env.sh`,  2) docker launch script in `host.sh` and 3) python training command in `train.sh` into all v5p workers, and starts the training afterwards. When all training steps completes, it will copy back the print out the average step time and copy back the profiling back under */tmp/home/profile/*. You shall see performance metric in the terminal as
```
[worker :x] Average step time: ...
```
that tells the average step time for each batch run of each worker.


### Environment Envs Explained

To make it simple, we suggest only change the following to env variables in env.sh:
*   `PER_HOST_BATCH_SIZE`:Batch size for each host/worker. High number can cause out of memory issue.
*   `TRAIN_STEPS`: How many training steps to run.
*   `PROFILE_DURATION`: Length of the profiling time (unit ms).
*   `RESOLUTION`: Image resolution.
