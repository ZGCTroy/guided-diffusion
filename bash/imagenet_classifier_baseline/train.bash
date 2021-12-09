CLASSIFIER_FLAGS="
    --classifier_pool attention --classifier_out_channels 200
    --classifier_image_size 256
    --classifier_num_channels 128 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 False --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "

#--data_dir /mnt/data1/shengming/ImageNetSubset/train
#    --val_data_dir /mnt/data1/shengming/ImageNetSubset/val

TRAIN_FLAGS="
    --iterations 300000 --anneal_lr True
    --batch_size 8 --lr 3e-4
    --save_interval 10000 --weight_decay 0.05
    --data_dir /mnt/disk50/datasets/ImageNet/train
    --val_data_dir /mnt/disk50/datasets/ImageNet/val
    --log_root /mnt/data1/shengming/guided_ddpm/log
    --save_name imagenet_classifier_baseline
    --tot_class 200 --dataset_type imagenet
    --gpus 2,3
    "

mpiexec -n 2 python classifier_train.py \
    $CLASSIFIER_FLAGS $TRAIN_FLAGS