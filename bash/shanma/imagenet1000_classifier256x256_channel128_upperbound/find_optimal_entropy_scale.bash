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


NNODE=2
#RANK=0
#MASTER_ADDR=localhost
#MASTER_PORT=23456


# 1
num_samples=5000
iteration=500000
steps=ddim25




  scale=1.0
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=1.25
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=1.5
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=1.75
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=2.0
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256


  scale=2.25
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=2.5
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256


for ((scale=5; scale<=10; scale=scale+1))
do


  predict_name=find_optimal_entropy_scale/model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion_uncond.pt --class_cond False \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim True --timestep_respacing ddim25

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \
  #      --use_ddim False --timestep_respacing 250

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256


done

steps=250

for ((scale=5; scale<=10; scale=scale+1))
do


  predict_name=find_optimal_entropy_scale/model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion_uncond.pt --class_cond False \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim False --timestep_respacing 250

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256


done



  scale=0.25
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim False --timestep_respacing 250

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=0.5
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim False --timestep_respacing 250

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256

  scale=0.75
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim False --timestep_respacing 250

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256


  scale=1.0
  predict_name=find_optimal_entropy_scale/conditional_model${iteration}_imagenet1000_steps${steps}_sample${num_samples}_scale${scale}_entropyScale
  python -m torch.distributed.launch \
           --nnode ${NNODE} --node_rank $RANK --nproc_per_node 8 \
         --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
         classifier_sample.py \
         $DIFFUSION_FLAGS $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS \
        --log_root ${logdir} \
        --model_path ${pretrain_model}/256x256_diffusion.pt --class_cond True \
        --classifier_path ${pretrain_model}/256x256_classifier.pt \
        --save_name ${save_name}/predict/${predict_name}\
        --classifier_scale ${scale} --num_samples ${num_samples} --batch_size 24 \
        --t_range_start 0 --t_range_end 1000 \
        --use_entropy_scale True \
        --use_ddim False --timestep_respacing 250

  #      --use_normalized_entropy_scale False
  #      --use_probability_scale True
  #      --selected_class 0 \
  #      --use_cond_range_scale True
  #      --gpus 0,1,2,3,4,5,6,7 \

  python evaluations/evaluator.py \
          --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
          --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
          --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
          --batch_size 256



