# Instructions for training Stable Diffusion 2.1 on TPU v5p

This documents present steps to run StableDiffusion [MaxDiffusion](https://github.com/google/maxdiffusion/tree/main/src/maxdiffusion) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

Login to gcloud, set XPK cluster project and zone.

```
gcloud auth application-default login
gcloud config set project <project_id>
gcloud config set compute/zone <cluster_zone>
```

Build a local docker image.

```
git clone -b mlperf_4 https://github.com/google/maxdiffusion.git
cd maxdiffusion
LOCAL_IMAGE_NAME=maxdiffusion_base_image
docker build  --no-cache --network host -f ./maxdiffusion.Dockerfile -t ${LOCAL_IMAGE_NAME} .
```

Install XPK and create GKE cluster.

```
pip install xpk
xpk cluster create --cluster <cluster_name> --tpu-type=<tpu_type> --num-slices=<num_slices>
```

Run workload using xpk.

```
export BASE_OUTPUT_DIR=gs://output_bucket/
DATA_DIR=gs://jfacevedo-maxdiffusion/laion400m/raw_data/tf_records_512_encoder_state_fp32
COMMITS=eac9132ef8b1a977372e29720fabc478529cd364

xpk workload create \
--cluster <cluster_name> \
--base-docker-image maxdiffusion_base_image \
--workload ${USER}-sd21-v5p \
--tpu-type=<tpu_type> \
--num-slices=<num_slices>  \
--command "cd maxdiffusion && git checkout $COMMITS && pip install . && python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name=<experiment_run_name> base_output_directory=${BASE_OUTPUT_DIR} \
train_data_dir=${DATA_DIR} per_device_batch_size=2 split_head_dim=True  attention=flash  train_new_unet=true norm_num_groups=16 \
start_step_to_checkpoint=5120000 enable_profiler=true skip_first_n_steps_for_profiler=5 reuse_example_batch=false max_train_steps=100
```

Clean up the GKE cluster.

```
xpk cluster delete --cluster <cluster_name>
```
