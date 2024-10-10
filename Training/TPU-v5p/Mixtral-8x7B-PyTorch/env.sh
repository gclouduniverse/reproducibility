# Uncomment below to set the Huggingface token
# HF_TOKEN=hf_***
PJRT_DEVICE=TPU
XLA_IR_DEBUG=1
XLA_HLO_DEBUG=1
PROFILE_EPOCH=0
PROFILE_STEP=3
PROFILE_DURATION_MS=120000
XLA_USE_SPMD=1
MAX_STEPS=20
SEQ_LENGTH=4096

# Per-host batch size is the number of training examples used by a TPU VM
# in each training step. For v5p, it will be 4 times the per-device batch size,
# since each TPU VM is connected to 4 v5p TPU chips. The following will lead
# to a per-device batch size of 8. Customize accordingly.
PER_HOST_BATCH_SIZE=32
