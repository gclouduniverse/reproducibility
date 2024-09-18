BASE_OUTPUT_DIR=$1
COMMITS=$2

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
rm -rf maxdiffusion

git clone https://github.com/google/maxdiffusion.git
cd maxdiffusion
git checkout ${COMMITS} 

python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 resolution=1024 per_device_batch_size=1 output_dir=$BASE_OUTPUT_DIR jax_cache_dir=${BASE_OUTPUT_DIR}/cache_dir/ max_train_steps=5000 attention=flash run_name=sdxl-fsdp-v5p-64-ddp