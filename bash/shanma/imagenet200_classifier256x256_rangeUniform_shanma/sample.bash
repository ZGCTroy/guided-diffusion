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
    --classifier_scale 10.0
    --classifier_pool attention --classifier_out_channels 200
    --classifier_image_size 256
    --classifier_num_channels 64 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 True --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "


SAMPLE_FLAGS="
    --batch_size 8 --num_samples 1000
    --use_ddim True --t_range_start 0 --t_range_end 750
    --log_root /workspace/guided-diffusion/log
    --gpus 0,1
    "

save_name="imagenet200_classifier256x256_channel64_rangeUniform"
logdir="/workspace/guided-diffusion/log"
pretrain_model="/workspace/guided-diffusion/pretrain_model"

for ((i=290000; i>=290000; i=i-30000))
do
    t=$(printf "%06d" $i)

    echo model${t}.pt

mpiexec -n 2 --allow-run-as-root python classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --model_path ${pretrain_model}/256x256_diffusion_uncond.pt \
      --classifier_path ${logdir}/${save_name}/model${t}.pt \
      --save_name ${save_name}/predict/model${t}_condrange750_scale2.0 \
      --classifier_scale 10.0 --num_samples 1000 --batch_size 12
done