TRAIN_FLAGS="
    --iterations 300000 --anneal_lr True
    --batch_size 16 --microbatch 16 --lr 1e-4
    --save_interval 10000 --weight_decay 0.05
    --data_dir /workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/train
    --val_data_dir /workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/val
    --log_root /workspace/guided-diffusion/log
    --tot_class 200
    --dataset_type imagenet-200 --imagenet200_class_list_file_path /workspace/guided-diffusion/datasets/list_tiny_imagenet.txt
    --gpus 0,1

    --only_projection False
    --schedule_sampler noisy-aware-weight --t_range_start 0 --t_range_end 1000
    "

CLASSIFIER_FLAGS="
    --classifier_pool attention --classifier_out_channels 200
    --classifier_image_size 256
    --classifier_num_channels 64 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
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
    --num_classes 200
    "

EVALUATE_FLAGS="
    --ref_batch /workspace/guided-diffusion/pretrain_model/VIRTUAL_tinyimagenet256_labeled.npz
    "

mpiexec -n 2 --allow-run-as-root python classifier_train.py \
    $CLASSIFIER_FLAGS $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $MODEL_FLAGS $EVALUATE_FLAGS \
     --save_name imagenet200_classifier256x256_channel64_NoisyImportance
#    --model_path /workspace/guided-diffusion/pretrain_model/256x256_diffusion_uncond.pt \
#    --classifier_path /workspace/guided-diffusion/pretrain_model/256x256_classifier.pt \
