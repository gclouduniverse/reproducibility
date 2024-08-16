# Instructions for training Mixtral-8X7B Maxtext on TPU v5p

This documents present steps to run Mixtral-8x7B [MaxText](https://github.com/google/maxtext) workload through [XPK](https://github.com/google/xpk/blob/main/README.md) tool.

Login to gcloud, set XPK cluster project and zone.

```
gcloud auth application-default login
gcloud config set project <project_id>
gcloud config set compute/zone <cluster_zone>
```

Build a local docker image with default name `maxtext_base_image`.

```
git clone https://github.com/google/maxtext.git
cd maxtext
bash docker_build_dependency_image.sh DEVICE=tpu
```

Install XPK and create GKE cluster.

```
pip install xpk
xpk cluster create --cluster <cluster_name> --tpu-type=<tpu_type> --num-slices=<num_slices>
```

Run workload in the maxtext github root directory.

```
export BASE_OUTPUT_DIR=gs://output_bucket/

xpk workload create \
--cluster <cluster_name> \
--base-docker-image maxtext_base_image \
--workload ${USER}-mixtral-8x7b \
--tpu-type=<tpu_type> \
--num-slices=<num_slices>  \
--command "python3 MaxText/train.py MaxText/configs/base.yml run_name=<experiment_run_name> per_device_batch_size=12 model_name=mixtral-8x7b steps=10 dtype=bfloat16 weight_dtype=bfloat16 max_target_length=4096 attention=flash dataset_type=synthetic tokenizer_path=assets/tokenizer.mistral"
```

Clean up the GKE cluster.

```
xpk cluster delete --cluster <cluster_name>
```
