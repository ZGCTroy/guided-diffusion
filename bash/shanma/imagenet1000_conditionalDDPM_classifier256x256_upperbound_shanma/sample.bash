DIFFUSION_FLAGS="
    --diffusion_steps 1000 --timestep_respacing ddim25
    --noise_schedule linear
    --learn_sigma True
"

MODEL_FLAGS="
    --learn_sigma True --class_cond True
    --image_size 256
    --num_channels 256 --num_res_blocks 2 --num_head_channels 64
    --attention_resolutions 32,16,8
    --use_fp16 True --resblock_updown True
    --use_scale_shift_norm True
    --num_classes 1000
    "

CLASSIFIER_FLAGS="
    --classifier_scale 2.5
    --classifier_pool attention --classifier_out_channels 1000
    --classifier_image_size 256
    --classifier_num_channels 128 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 True --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "


SAMPLE_FLAGS="
    --batch_size 8 --num_samples 1000
    --use_ddim True --t_range_start 0 --t_range_end 1000
    "

workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
save_name="imagenet1000_conditionalDDPM_classifier256x256_channel128_upperbound"
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model

for ((i=500000; i>=500000; i=i-30000))
do
    t=$(printf "%06d" $i)

    echo model${t}.pt

mpiexec -n 8 --allow-run-as-root python classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --gpus 0,1,2,3,4,5,6,7 \
      --model_path ${pretrain_model}/256x256_diffusion.pt \
      --classifier_path ${pretrain_model}/256x256_classifier.pt \
      --save_name ${save_name}/predict/model${t}_imagenet1000_stepsddim25_sample50000 \
      --classifier_scale 2.5 --num_samples 50000 --batch_size 24
#      --use_entropy_scale True
#      --use_ddim False --timestep_respacing 250


mpiexec -n 8 --allow-run-as-root python classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --gpus 0,1,2,3,4,5,6,7 \
      --model_path ${pretrain_model}/256x256_diffusion.pt \
      --classifier_path ${pretrain_model}/256x256_classifier.pt \
      --save_name ${save_name}/predict/model${t}_imagenet1000_stepsddim25_sample50000_entropyScale \
      --classifier_scale 2.5 --num_samples 50000 --batch_size 24 \
      --use_entropy_scale True
#      --use_ddim False --timestep_respacing 250

done


