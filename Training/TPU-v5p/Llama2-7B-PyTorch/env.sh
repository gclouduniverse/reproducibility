# Uncomment below to set the Huggingface token
# HF_TOKEN=hf_***

DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch/docker/reproducibility/llama3:v0
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
