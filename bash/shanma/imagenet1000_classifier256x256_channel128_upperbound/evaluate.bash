workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model

save_name="imagenet1000_classifier256x256_channel128_upperbound"
num_samples="50000"
scale="10.0"
steps="ddim25"

for ((i=500000; i>=500000; i=i-30000))
do
    t=$(printf "%06d" $i)

    echo $t

  predict_name=model500000_imagenet1000_stepsddim25_sample50000_entropyScaleRange0-500

  CUDA_VISIBLE=0,1 python evaluations/evaluator.py \
        --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
        --sample_batch ${logdir}/${save_name}/predict/${predict_name}/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
        --save_result_path ${logdir}/${save_name}/predict/${predict_name}/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
        --batch_size 256


done