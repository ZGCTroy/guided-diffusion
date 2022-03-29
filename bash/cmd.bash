#tensorboard --logdir ./log --bind_all --max_reload_threads 10 --load_fast=false

cd /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion

rm -rf /opt/conda/lib/python3.8/site-packages/guided-diffusion

python setup.py build develop

nvidia-smi

ls


# 256x256
model_name=imagenet1000_classifier256x256_channel128_MCD
/bin/bash ./bash/shanma/${model_name}/train.bash

/bin/bash ./bash/shanma/${model_name}/sample.bash
