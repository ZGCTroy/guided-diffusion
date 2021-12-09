DIFFUSION_FLAGS="
    --diffusion_steps 1000
    --noise_schedule linear
    --learn_sigma True
"

MODEL_FLAGS="
    --model_path /mnt/data1/shengming/guided_ddpm/pretrain/256x256_diffusion_uncond.pt
    --learn_sigma True --class_cond False
    --image_size 256
    --num_channels 256 --num_res_blocks 2 --num_head_channels 64
    --attention_resolutions 32,16,8
    --use_fp16 False --resblock_updown True
    --use_scale_shift_norm True
    --num_classes 200
    "

#c
#--classifier_path /mnt/data1/shengming/guided_ddpm/log/imagenet_classifier_finetune/model000019.pt

CLASSIFIER_FLAGS="
    --classifier_scale 10.0
    --classifier_path /mnt/data1/shengming/guided_ddpm/pretrain/256x256_classifier.pt
    --classifier_pool attention --classifier_out_channels 200
    --classifier_image_size 256
    --classifier_num_channels 256 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 False --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "

SAMPLE_FLAGS="
    --batch_size 4 --num_samples 40
    --timestep_respacing 50 --use_ddim False
    --log_root /mnt/data1/shengming/guided_ddpm/log
    --save_name imagenet_classifier_finetune/predict/test0
    --gpus 0,1
    "

mpiexec -n 2 python classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS

