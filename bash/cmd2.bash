#tensorboard --logdir ./log --bind_all --max_reload_threads 10 --load_fast=false

cd /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion

rm -rf /opt/conda/lib/python3.8/site-packages/guided-diffusion

python setup.py build develop

nvidia-smi

ls


# 64x64
model_name=imagenet1000_classifier64x64_channel128_entropyConstraintTrain0.05
/bin/bash ./bash/shanma/${model_name}/train.bash
