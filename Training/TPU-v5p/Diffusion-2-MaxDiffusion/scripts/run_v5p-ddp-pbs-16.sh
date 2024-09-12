DATA_DIR=$1
BASE_OUTPUT_DIR=$2
COMMITS=$3

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
done

export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_megacore_fusion=false --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true'

LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true --xla_enable_async_all_reduce=true'
LIBTPU_INIT_ARGS+=' --xla_tpu_enable_async_collective_fusion_with_mosaic_custom_call=true --xla_tpu_mosaic_fusion=true'
LIBTPU_INIT_ARGS+=' --xla_enable_async_reduce_scatter_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=true'
LIBTPU_INIT_ARGS+=' --xla_tpu_spmd_threshold_for_allgather_cse=1000000 --xla_jf_spmd_threshold_for_windowed_einsum_mib=1000000'

#reload code to specific commits
cd maxdiffusion
git checkout ${COMMITS}
pip install .

python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name=sd_base2 base_output_directory=${BASE_OUTPUT_DIR} \
train_data_dir=${DATA_DIR} per_device_batch_size=16 split_head_dim=True  attention=flash  train_new_unet=true norm_num_groups=16 \
dcn_data_parallelism=${NUM_SLICES} \
start_step_to_checkpoint=5120000 enable_profiler=true skip_first_n_steps_for_profiler=5 reuse_example_batch=false max_train_steps=100