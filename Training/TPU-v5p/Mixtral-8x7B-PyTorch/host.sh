#!/bin/bash

DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral@sha256:1e9024d13e53bdadc13d7a695d8fbe52b95a78cfae2a101da8a9d8fc94b1c96b

worker_id=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number" -H 'Metadata-Flavor: Google')

cat >> /dev/null <<EOF
EOF

stdbuf -oL bash <<-PIPE_EOF 2>&1 | sed "s/^/[worker $slice_id:$worker_id] /g" | tee runlog
  set -o xtrace
  # Configure docker
  sudo groupadd docker
  sudo usermod -aG docker $USER
  # newgrp applies updated group permissions
  newgrp - docker
  gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
  # Kill any running benchmarks
  docker kill $USER-test
  docker pull $DOCKER_IMAGE
  docker run --rm \
    --name $USER-test \
    --privileged \
    --env-file env.sh \
    -v /home/$USER:/tmp/home \
    --shm-size=16G \
    --net host \
    -u root \
    --entrypoint /bin/bash $DOCKER_IMAGE \
    /tmp/home/train.sh

PIPE_EOF