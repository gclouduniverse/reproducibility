# Uncomment below to set the Huggingface token
# HF_TOKEN=hf_***

DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/mixtral@sha256:1e9024d13e53bdadc13d7a695d8fbe52b95a78cfae2a101da8a9d8fc94b1c96b
PJRT_DEVICE=TPU
XLA_IR_DEBUG=1
XLA_HLO_DEBUG=1
PROFILE_EPOCH=0
PROFILE_STEP=3
PROFILE_DURATION_MS=120000
XLA_USE_SPMD=1
MAX_STEPS=20
SEQ_LENGTH=4096
BATCH_SIZE=512
