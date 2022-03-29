CUDA_VISIBLE_DEVICES=0,1 python evaluations/evaluator.py \
        --ref_batch /mnt/data1/shengming/guided_ddpm/pretrain/VIRTUAL_imagenet256_labeled.npz \
        --sample_batch /mnt/data1/shengming/guided_ddpm/pretrain/admnet_imagenet256.npz



# python evaluations/evaluator_pytorch.py \
#     --path_real /mnt/data1/shengming/guided_ddpm/pretrain/VIRTUAL_imagenet256_labeled.npz \
# --path_fake /mnt/data1/shengming/guided_ddpm/pretrain/admnet_imagenet256.npz
