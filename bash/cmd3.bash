#tensorboard --logdir ./log --bind_all --max_reload_threads 10 --load_fast=false

cd /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion

nvidia-smi

ls



model_name=imagenet1000_classifier64x64_channel128_baseline

/bin/bash ./bash/shanma/${model_name}/find_optimal_entropy_scale.bash


model_name=imagenet1000_classifier128x128_channel128_baseline

/bin/bash ./bash/shanma/${model_name}/find_optimal_entropy_scale.bash