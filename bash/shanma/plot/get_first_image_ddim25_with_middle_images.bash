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
    --use_entropy_scale False
    "

workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
save_name="get_first_image_ddim25_with_middle_images"
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model
iteration=500000
steps='ddim25'
num_samples=50000

WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=23456
batch_size=24



 # 金鱼
 steps=ddim25
    scale=10.0
    predict_name=model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 1 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample_to_get_middle_image_v2.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/256x256_diffusion_uncond.pt --class_cond False \
      --classifier_path ${pretrain_model}/256x256_classifier.pt \
      --use_entropy_scale False \
      --use_ddim True --timestep_respacing ${steps} \
      --selected_class 970


 steps=ddim25
    scale=10.0
    predict_name=model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 1 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample_to_get_middle_image_v2.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/256x256_diffusion_uncond.pt --class_cond False \
      --classifier_path ${pretrain_model}/256x256_classifier.pt \
      --use_entropy_scale False \
      --use_ddim True --timestep_respacing ${steps} \
      --selected_class 984

 steps=ddim25
    scale=10.0
    predict_name=model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 1 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample_to_get_middle_image_v2.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/256x256_diffusion_uncond.pt --class_cond False \
      --classifier_path ${pretrain_model}/256x256_classifier.pt \
      --use_entropy_scale False \
      --use_ddim True --timestep_respacing ${steps} \
      --selected_class 985