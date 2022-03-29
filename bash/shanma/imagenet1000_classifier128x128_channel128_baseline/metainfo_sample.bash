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
    --num_classes 1000
    "

CLASSIFIER_FLAGS="
    --classifier_scale 10.0
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
save_name="imagenet1000_classifier256x256_channel128_upperbound"
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model


NNODE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=23456



num_samples=1000
predict_name=model500000_imagenet1000_stepsddim25_sample${num_samples}_plot_entropyScale
scale=5.0
steps=ddim25

python -m torch.distributed.launch \
         --nnode ${NNODE} --node_rank $RANK --nproc_per_node 2 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample_for_plot.py \
       $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --model_path ${pretrain_model}/256x256_diffusion_uncond.pt \
      --classifier_path ${pretrain_model}/256x256_classifier.pt \
      --save_name ${save_name}/predict/${predict_name}\
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
      --use_entropy_scale True

#      --selected_class 0
#      --t_range_start 500 --t_range_end 1000
#      --use_probability_scale True
#      --use_cond_range_scale True
#      --gpus 0,1,2,3,4,5,6,7 \
#      --use_ddim False --timestep_respacing 250



