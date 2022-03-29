DIFFUSION_FLAGS="
    --diffusion_steps 1000 --timestep_respacing ddim25
    --noise_schedule linear
    --learn_sigma True
"

MODEL_FLAGS="
    --learn_sigma True --class_cond True
    --image_size 128
    --num_channels 256 --num_res_blocks 2
    --attention_resolutions 32,16,8
    --use_fp16 True --resblock_updown True
    --use_scale_shift_norm True
    --num_classes 1000
    --num_heads 4
    "
#--num_head_channels 64
#    --num_heads 4

CLASSIFIER_FLAGS="
    --classifier_pool attention --classifier_out_channels 1000
    --classifier_image_size 128
    --classifier_num_channels 128 --classifier_num_res_blocks 2 --classifier_num_head_channels 64
    --classifier_attention_resolutions 32,16,8
    --classifier_use_fp16 True --classifier_resblock_updown True
    --classifier_use_scale_shift_norm True
    "


SAMPLE_FLAGS="
    --t_range_start 0 --t_range_end 1000
    "

workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
save_name="imagenet1000_classifier128x128_channel128_baseline"
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model
iteration=300000
num_samples=50000
batch_size=72

WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=23456

#    steps='ddim25'
#    scale=1.25
#    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}
#
#    python -m torch.distributed.launch \
#       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
#       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#       classifier_sample.py \
#      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
#      --log_root ${logdir} \
#      --save_name ${save_name}/predict/${predict_name} \
#      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
#      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
#      --classifier_path ${pretrain_model}/128x128_classifier.pt \
#      --use_ddim True --timestep_respacing ${steps} \
#      --use_entropy_scale False
#
#    python evaluations/evaluator.py \
#          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
#          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
#          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
#          --batch_size 256

#    steps='250'
#    scale=0.5
#    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}
#
#    python -m torch.distributed.launch \
#       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
#       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
#       classifier_sample.py \
#      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
#      --log_root ${logdir} \
#      --save_name ${save_name}/predict/${predict_name} \
#      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
#      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
#      --classifier_path ${pretrain_model}/128x128_classifier.pt \
#      --use_ddim False --timestep_respacing ${steps} \
#      --use_entropy_scale False
#
#    python evaluations/evaluator.py \
#          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
#          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
#          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
#          --batch_size 256

    steps='ddim25'
    scale=1.5
    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
      --classifier_path ${pretrain_model}/128x128_classifier.pt \
      --use_ddim True --timestep_respacing ${steps} \
      --use_entropy_scale False

    python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

    steps='ddim25'
    scale=1.2
    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
      --classifier_path ${pretrain_model}/128x128_classifier.pt \
      --use_ddim True --timestep_respacing ${steps} \
      --use_entropy_scale False

    python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

    steps='ddim25'
    scale=1.1
    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
      --classifier_path ${pretrain_model}/128x128_classifier.pt \
      --use_ddim True --timestep_respacing ${steps} \
      --use_entropy_scale False

    python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

    steps='ddim25'
    scale=1.0
    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
      --classifier_path ${pretrain_model}/128x128_classifier.pt \
      --use_ddim True --timestep_respacing ${steps} \
      --use_entropy_scale False

    python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

    steps='ddim25'
    scale=0.75
    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
      --classifier_path ${pretrain_model}/128x128_classifier.pt \
      --use_ddim True --timestep_respacing ${steps} \
      --use_entropy_scale False

    python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

    steps='ddim25'
    scale=0.5
    predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_sample${num_samples}

    python -m torch.distributed.launch \
       --nnode $WORLD_SIZE --node_rank $RANK --nproc_per_node 8 \
       --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
       classifier_sample.py \
      $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
      --log_root ${logdir} \
      --save_name ${save_name}/predict/${predict_name} \
      --classifier_scale ${scale} --num_samples ${num_samples} --batch_size ${batch_size} \
      --model_path ${pretrain_model}/128x128_diffusion.pt --class_cond True \
      --classifier_path ${pretrain_model}/128x128_classifier.pt \
      --use_ddim True --timestep_respacing ${steps} \
      --use_entropy_scale False

    python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet128_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x128x128x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256







