#!/bin/bash

python /workspace/diffusers/examples/text_to_image/train_text_to_image_xla.py \
--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-base \
--dataset_name=$DATASET_NAME --resolution=$RESOLUTION --center_crop --random_flip \
--train_batch_size=$PER_HOST_BATCH_SIZE --max_train_steps=$TRAIN_STEPS \
--learning_rate=1e-06 --mixed_precision=bf16 --profile_duration=$PROFILE_DURATION \
--output_dir=$OUTPUT_DIR --dataloader_num_workers=4 \
--loader_prefetch_size=4 --device_prefetch_size=4
