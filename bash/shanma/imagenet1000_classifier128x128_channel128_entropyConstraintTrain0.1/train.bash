TRAIN_FLAGS="
    --iterations 310000 --batch_size 16 --microbatch 16
    --anneal_lr True --lr 3e-4 --weight_decay 0.05
    --save_interval 10000
    --log_interval 500 --eval_interval=500 --save_interval 10000

    --data_dir /workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/train
    --val_data_dir /workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/val
    --log_root /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log

    --tot_class 1000
    --dataset_type imagenet-1000
    --imagenet200_class_list_file_path /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/datasets/list_tiny_imagenet.txt

    --schedule_sampler uniform --t_range_start 0 --t_range_end 1000
    "

CLASSIFIER_FLAGS="
    --classifier_pool attention --classifier_out_channels 1000
    --classifier_image_size 128
    --classifier_num_channels 128 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 True --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "

DIFFUSION_FLAGS="
    --diffusion_steps 1000
    --noise_schedule linear
    --learn_sigma True
"


save_name="imagenet1000_classifier128x128_channel128_entropyConstraintTrain0.1"

#WORLD_SIZE=1
#RANK=0
#MASTER_ADDR=localhost
#MASTER_PORT=23456

python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_train.py \
    $CLASSIFIER_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $EVALUATE_FLAGS \
     --save_name $save_name \
     --uncertainty_lambda=0.1 --use_uncertainty_loss True