DIFFUSION_FLAGS="
    --diffusion_steps 1000 --timestep_respacing ddim25
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

CLASSIFIER_FLAGS="
    --classifier_scale 2.5
    --classifier_pool attention --classifier_out_channels 200
    --classifier_image_size 256
    --classifier_num_channels 64 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 True --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "


SAMPLE_FLAGS="
    --batch_size 16 --num_samples 1000
    --use_ddim True
    --log_root /workspace/guided-diffusion/log
    --gpus 0,1
    "

mpiexec -n 2 --allow-run-as-root python classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --model_path /workspace/guided-diffusion/pretrain_model/256x256_diffusion_uncond.pt \
      --classifier_path /workspace/guided-diffusion/log/imagenet200_classifier256x256_channel64_baseline/model010000.pt \
      --save_name imagenet200_classifier256x256_channel64_baseline/predict/model010000

