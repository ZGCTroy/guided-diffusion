#CLASSIFIER_FLAGS="
#    --classifier_pool attention --classifier_out_channels 100
#    --classifier_image_size 256
#    --classifier_num_channels 64 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
#    --classifier_attention_resolutions 32,16,8
#    --classifier_use_fp16 True --classifier_resblock_updown True
#    --classifier_use_scale_shift_norm True
#    "
#
##--data_dir /mnt/data1/shengming/ImageNetSubset/train
##--val_data_dir /mnt/data1/shengming/ImageNetSubset/val
##--data_dir /mnt/disk50/datasets/ImageNet/train
##    --val_data_dir /mnt/disk50/datasets/ImageNet/val
#
#TRAIN_FLAGS="
#    --iterations 300000 --anneal_lr True
#    --batch_size 16 --lr 3e-4
#    --save_interval 50 --weight_decay 0.05
#    --log_interval 100
#    --evaluate_between_train True
#    --eva_timestep_respacing ddim25
#
#    --data_dir /mnt/data2/shengming/ImageNet/train
#    --val_data_dir /mnt/data2/shengming/ImageNet/val
#
#    --log_root /mnt/data1/shengming/guided_ddpm/log
#    --save_name debug
#    --tot_class 100 --dataset_type imagenet-1000
#    --gpus 2,3
#    "
#
#mpiexec -n 1 python classifier_train.py \
#    $CLASSIFIER_FLAGS $TRAIN_FLAGS


rsync -avzP -e "ssh -i $HOME/.ssh/id_rsa"  root@ssh.atom.ks.supremind.info:51324:/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2/predict/model500000_stepsddim25_scale6.0_sample50000_entropyScale ../visualization/