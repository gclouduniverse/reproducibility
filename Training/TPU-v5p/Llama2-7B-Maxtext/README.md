# Instructions for training Llama2-7B-Maxtext on TPU v5p

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
export CLUSTER_NAME=v4-demo #<your_cluster_name>
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v4-128 #<your TPU Type>
export NUM_SLIECES=1 #<number of TPU node-pools you want to create>
```

2. Create the network and firewall for this cluster if it doesn’t exist yet.
```
gcloud compute networks create ${NETWORK_NAME} --mtu=8896 --project=${PROJECT} --subnet-mode=auto --bgp-routing-mode=regional
gcloud compute firewall-rules create ${NETWORK_FW_NAME} --network ${NETWORK_NAME} --allow tcp,icmp,udp --project=${PROJECT}
```

3. Create GKE cluster with TPU node-pools
```
python3 xpk.py cluster create \
--default-pool-cpu-machine-type=n1-standard-32 \
--cluster ${CLUSTER_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLIECES} \
--custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
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
--num-slices=${NUM_SLIECES} \
--command "echo Hello World"
```
5. You can also check your workload status with the following command:
  ```
python3 xpk.py workload list \
--cluster ${CLUSTER_NAME}
  ```
6. For more information about XPK, please refer to this [link](https://github.com/google/xpk).

## Run Maxtext Llama2-7B workloads on GKE
1. Clone [Maxtext](https://github.com/google/maxtext) repo
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

4. Specify your workload configs
```
export CLUSTER_NAME=v4-demo #<your_cluster_name>
export WORKLOAD_NAME=llam2-7b-test #<your_workload_name>
export TPU_TYPE=v4-128 #<your TPU Type>
export NUM_SLIECES=1 #<number of TPU node-pools you want to use>
export LOCAL_IMAGE_NAME=gcr.io/${PROJECT}/${USER}_runner
export OUTPUT_PATH=gs://v4-demo/ #<your_GCS_folder_for_results>
```
* You should be able to to see results like this: ![image](https://github.com/user-attachments/assets/c33010a6-e109-411e-8fb5-afb4edb3fa72)


5. Switch back to your XPK folder and run Llama2-7B workload
```
cd ../ #Make sure you are in the XPK folder

python3 xpk.py workload create \
--cluster ${CLUSTER_NAME} \
--workload ${WORKLOAD_NAME} \
--tpu-type=${TPU_TYPE} \
--num-slices=${NUM_SLIECES} \
--docker-image=${LOCAL_IMAGE_NAME} \
--command "\
   python MaxText/train.py MaxText/configs/base.yml\
   model_name=llama2-7b\
   base_output_directory=$OUTPUT_PATH\
   dataset_type=synthetic\
   tokenizer_path=assets/tokenizer.llama2\
   per_device_batch_size=16\
   enable_checkpointing=false\
   gcs_metrics=true\
   profiler=xplane\
   skip_first_n_steps_for_profiler=5\
   steps=10"
```
* You should see the output similar to this: ![image](https://github.com/user-attachments/assets/8ffa72cb-61b1-4f87-a01e-f80d2330341a)
* Here is an example of the output for your GCS folder: ![image](https://github.com/user-attachments/assets/e6a5d808-d401-4854-9630-ad79bccd3044)

6. [Optional] If you need to delete any of your workload, you can run the following command:
```
export WORKLOAD_NAME_TO_DELETE=llam2-7b-test

python3 xpk.py workload delete \
--workload ${WORKLOAD_NAME_TO_DELETE} \
--cluster ${CLUSTER_NAME}
```

## GKE Cluster Deletion
You can use the following command to delete GKE cluster:
```
export CLUSTER_NAME=v4-demo #<your_cluster_name>

python3 xpk.py cluster delete \
--cluster $CLUSTER_NAME
```


