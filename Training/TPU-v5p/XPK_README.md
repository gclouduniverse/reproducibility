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
export NETWORK_NAME=${CLUSTER_NAME}-only-mtu9k
export NETWORK_FW_NAME=${NETWORK_NAME}-only-fw
export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"
export TPU_TYPE=v5p-512 #<your TPU Type>
export NUM_SLICES=1 #<number of TPU node-pools you want to create>
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
--num-slices=${NUM_SLICES} \
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

## GKE Cluster Deletion
You can use the following command to delete GKE cluster:
```
export CLUSTER_NAME=v5p-demo #<your_cluster_name>

python3 xpk.py cluster delete \
--cluster $CLUSTER_NAME
```