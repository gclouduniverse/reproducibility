# Instructions for training GPT3-175B NeMo on A3Mega

## Setup

Clone repository


```shell
git clone git@github.com:gclouduniverse/reproducibility.git
```

### Google Cloud

Login to gcloud, set project:

```shell
gcloud auth application-default login
gcloud config set project <project_id>
```

Get cluster credentials for kubeconfig:

```shell
gcloud container clusters get-credentials <cluster_name> --zone <cluster_zone>

# Example
gcloud container clusters get-credentials a3plus-benchmark --zone australia-southeast1
```

If using gcloud, register gcloud as a Docker credential helper:

```shell
gcloud auth configure-docker <registries>

# Example
gcloud auth configure-docker us-docker.pkg.dev
```

### Helm

To install Helm:


```
$ curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
$ chmod 700 get_helm.sh
$ ./get_helm.sh
```


Set helm binary to be executable:


```
sudo chmod +x /usr/local/bin/helm 
```


Update values in /helm_context/values.yaml.


- You **must** modify the workload.image string value to match the one built earlier. Look for "EDIT THIS".
- You **must** modify the workload.gcsBucketForDataCataPath string value to match desired existing bucket. This bucket must be in the same region as the cluster. Look for "EDIT THIS".
- You may also settings like the number of gpus or which workload image to use. 


```
vi helm-context/values.yaml
```


Update values in workload configuration


```
vi nemo-configurations/gpt3-175b-16gpus-bf16.yaml
```

## Training

To run GPT3-175B-NeMo:

```shell
cd /path/to/reproducibility/Training/A3Mega/GPT3-175B-NeMo
```

### Docker Image

Build and push docker image:

```shell
cd docker
docker build -t <regristry_path_image_name>:<image_tag> -f nemo_example.Dockerfile .
docker push <registry_path_image_name>:<image_tag>
cd ..

# Example
cd docker
docker build -t us-east4-docker.pkg.dev/supercomputer-testing/reproducibility/nemo_test:24.05 -f nemo_example.Dockerfile .
docker push us-east4-docker.pkg.dev/supercomputer-testing/reproducibility/nemo_test:24.05
cd ..
```

### Run workflow

In order for this workflow to function, in the ```helm-context``` directory, there must exist a **_select-configuration.yaml_** file. This can be achieved by copying an appropriate file to this location.


```shell
cd helm-context
cp nemo-configurations/gpt3-175b-16gpus-bf16.yaml ./selected-configuration.yaml
```

Package and schedule job. An example job name could be "nemo-gpt3-175b-nemo-16gpus". Use whatever is convenient when searching for later.


```shell
helm install <username_workload_job_name> helm-context/

# Example
helm install nemo-gpt3-175b-nemo-16gpus helm-context/
```

### Monitor workflow

Check pod status (use this to find the name of the pod you want logs from)


```shell
kubectl get pods | grep "<some_part_of_username_workload_job_name>"
```


Check job status


```shell
kubectl get jobs | grep "<some_part_of_username_workload_job_name>"
```


Get logs (Using pod name from earlier)


```shell
kubectl logs "<pod_name>"
```


### Check results


Nsight logs located specified cloud storage


```
/path/to/your/bucket/<run_id>/rank-0.nsys-rep
```

Example


```
gs://benchmarks/nemo-experiments/nemo-gpt3-175b-nemo-16gpus-1721164368-fwuz/rank-0.nsys-rep
```
