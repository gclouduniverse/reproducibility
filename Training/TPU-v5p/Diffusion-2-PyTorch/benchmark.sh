#!/bin/bash
# SCP the environment setup to all instances. Used in `--env-file` in `docker run` on the host script.
gcloud compute tpus tpu-vm scp env.sh train.sh $TPU_NAME:~ --worker=all --project $PROJECT --zone=$ZONE

# Actually runs the benchmark.
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE --worker=all --command="$(cat host.sh)"

# Copy the profile and output back
gcloud compute tpus tpu-vm scp --recurse  $TPU_NAME:~/{profile,output} ./ --project=$PROJECT --zone=$ZONE
