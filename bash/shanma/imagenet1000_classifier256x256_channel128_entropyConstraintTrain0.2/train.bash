TRAIN_FLAGS="
    --iterations 510000 --anneal_lr True
    --batch_size 16 --microbatch 16 --lr 3e-4
    --save_interval 10000 --weight_decay 0.05

    --data_dir /workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/train
    --val_data_dir /workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/val
    --log_root /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log

    --tot_class 1000
    --dataset_type imagenet-1000
    --imagenet200_class_list_file_path /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/datasets/list_tiny_imagenet.txt

    --only_projection False
    --schedule_sampler uniform --t_range_start 0 --t_range_end 1000
    "

CLASSIFIER_FLAGS="
    --classifier_pool attention --classifier_out_channels 1000
    --classifier_image_size 256
    --classifier_num_channels 128 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 True --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "

SAMPLE_FLAGS="
    --sample_test_between_train False
    --eva_timestep_respacing ddim25
    --num_samples 512
    --use_ddim True
    --clip_denoised True
    --classifier_scale 10.0
    "
DIFFUSION_FLAGS="
    --diffusion_steps 1000
    --noise_schedule linear
    --learn_sigma True
"

MODEL_FLAGS="
    --learn_sigma True --class_cond False
    --image_size 256
    --num_channels 256 --num_res_blocks 2 --num_head_channels 64
    --attention_resolutions 32,16,8
    --use_fp16 True --resblock_updown True
    --use_scale_shift_norm True
    --num_classes 1000
    "

EVALUATE_FLAGS="
    --ref_batch /workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/pretrain_model/VIRTUAL_imagenet256_labeled.npz
    "

python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_train.py \
    $CLASSIFIER_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $EVALUATE_FLAGS \
     --save_name imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2 \
     --uncertainty_lambda=0.2 --uncertainty_variance_constrain=False --use_uncertainty_loss True