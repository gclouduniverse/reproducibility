# Instructions for training GPT3-175B-Maxtext on TPU v5p

## Initialization
1. Run the following commands to initialize the project and zone.
```
export PROJECT=tpu-prod-env-multipod #<your_project_id>
export ZONE=us-central2-b #<zone>
gcloud config set project $PROJECT
gcloud config set compute/zone $ZONE
```

2. Clone XPK repo.
```
git clone https://github.com/google/xpk.git
cd xpk
```

## GKE Cluster Creation 
1. Specify your TPU GKE cluster configs.
```
export CLUSTER_NAME=v5p-demo #<your_cluster_name>
export TPU_TYPE=v5p-512 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>
```


2. Create GKE cluster with TPU node-pools
```
python3 xpk.py cluster create \
--cluster ${CLUSTER_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--on-demand
```

  * Noted: TPU has `reserved`, `on-demand`, `spot` quota. This example used the `on-demand` quota. If you have the reserved or spot quota, please refer to this [link](https://github.com/google/xpk?tab=readme-ov-file#cluster-create).
  * If you want to check what quota you have, please refer to this [link](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus#ensure-quota).
  * You should be able to see your GKE cluster similar to this once it is created successfully:![image](https://github.com/user-attachments/assets/60743411-5ee5-4391-bb0e-7ffba4d91c1d)


4. Test your GKE cluster to make sure it is usable
```
python3 xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--workload hello-world-test \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--command "echo Hello World"
```
* You should be able to to see results like this: ![image](https://github.com/user-attachments/assets/c33010a6-e109-411e-8fb5-afb4edb3fa72)

5. You can also check your workload status with the following command:
  ```
python3 xpk.py workload list \
--cluster ${CLUSTER_NAME}
  ```
6. For more information about XPK, please refer to this [link](https://github.com/google/xpk).

## Run Maxtext GPT3-175B workloads on GKE
1. Clone [Maxtext](https://github.com/google/maxtext) repo and move to its directory
```
git clone https://github.com/google/maxtext.git
cd maxtext
```

2. Run the following commands to build the docker image
```
bash docker_build_dependency_image.sh MODE=stable DEVICE=tpu
```

3. Upload your docker image to Container Registry
```
bash docker_upload_runner.sh CLOUD_IMAGE_NAME=${USER}_runner
```

4. Create your GCS bucket
```
GCS_PATH=gs://v5p-demo #<your_GCS_folder_for_results>
gcloud storage buckets create ${GCS_PATH}  --project ${PROJECT}
```

5. Specify your workload configs
```
export CLUSTER_NAME=v5p-demo #<your_cluster_name>
export WORKLOAD_NAME=gpt3-175b-test #<your_workload_name>
export RUN_NAME=gpt3-175b-run #<your_run_name>
export TPU_TYPE=v5p-1024 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/${PROJECT}/${USER}_runner
export OUTPUT_PATH=gs://v5p-demo/ #<your_GCS_folder_for_results>
```

6. From the MaxText root directory, start your GPT3-175B workload
```

python3 ../xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLICES} \
--docker-image=${LOCAL_IMAGE_NAME} \
--command "python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME model_name=gpt3-175b base_output_directory=$OUTPUT_PATH enable_checkpointing=false async_checkpointing=false steps=20 per_device_batch_size=4 ici_data_parallelism=8 ici_fsdp_parallelism=8 ici_tensor_parallelism=8 remat_policy=full attention=flash quantization=int8 dataset_type=synthetic"
```

From your workload logs, you should start seeing step time logs like the following:
```
completed step: 2, seconds: 22.197, TFLOP/s/device: 397.246, Tokens/s/device: 369.059, total_weights: 4194304, loss: 0.000
```

7. [Optional] If you need to delete your workload, you can run the following command:
```
cd .. # Switch back to the xpk directory
export WORKLOAD_NAME_TO_DELETE=gpt3-175b-test

python3 xpk.py workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```

## GKE Cluster Deletion
You can use the following command to delete GKE cluster:
```
export CLUSTER_NAME=v5p-demo #<your_cluster_name>

python3 xpk.py cluster delete \
--cluster $CLUSTER_NAME
```