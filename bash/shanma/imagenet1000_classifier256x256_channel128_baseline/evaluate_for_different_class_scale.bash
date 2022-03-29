workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model

num_samples="50000"
scale="10.0"
steps="ddim25"
image_size=256
iteration=300000
save_name=imagenet1000_classifier${image_size}x${image_size}_channel128_baseline

predict_name=different_scale_for_different_class

CUDA_VISIBLE=0,1 python evaluations/evaluator_for_different_class_and_scale.py \
      --ref_batch ${pretrain_model}/VIRTUAL_imagenet-1000_256x256_labeled_sample50_class0-999.npz \
      --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result.yaml \
      --batch_size 512

#      --sample_batch ${pretrain_model}/VIRTUAL_imagenet${image_size}_labeled.npz \
#      --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x${image_size}x${image_size}x3.npz \


