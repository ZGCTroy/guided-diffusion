workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model

save_name="imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2"
num_samples="50000"
scale="1.0"
steps="250"
iteration=500000

predict_name=conditional_model${iteration}_steps${steps}_scale${scale}_entropyScale

CUDA_VISIBLE=0,1 python evaluations/evaluator.py \
      --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
      --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
      --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
      --batch_size 256


