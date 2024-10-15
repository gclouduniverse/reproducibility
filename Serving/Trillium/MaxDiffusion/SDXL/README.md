# Setup


## Step 1: Installing the dependencies:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc

conda create -n tpu python=3.10 
source activate tpu

https://github.com/google/maxdiffusion.git && cd maxdiffusion
git checkout mlperf4.1

pip install -e .
pip install -r requirements.txt
pip install -U --pre jax[tpu] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Step 2: Running the inference benchmark:

```
LIBTPU_INIT_ARGS="--xla_tpu_rwb_fusion=false --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_scoped_vmem_limit_kib=65536" python -m src.maxdiffusion.generate_sdxl src/maxdiffusion/configs/base_xl.yml run_name="my_run"
```


