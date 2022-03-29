workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model

save_name="imagenet200_classifier256x256_channel64_uncertainty0.2"
num_samples="10000"
scale="10.0"
steps="ddim25"

for ((i=290000; i>=290000; i=i-30000))
do
    t=$(printf "%06d" $i)

    echo $t

#    CUDA_VISIBLE=0,1 python evaluations/evaluator.py \
#      --ref_batch ${pretrain_model}/VIRTUAL_tinyimagenet256_labeled10000.npz \
#      --sample_batch ${logdir}/${save_name}/predict/model${t}_scale10.0/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
#      --save_result_path ${logdir}/${save_name}/predict/model${t}_scale10.0/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
#      --batch_size 256

    CUDA_VISIBLE=0,1 python evaluations/evaluator.py \
      --ref_batch ${pretrain_model}/VIRTUAL_tinyimagenet256_labeled10000.npz \
      --sample_batch ${logdir}/${save_name}/predict/model${t}_scale10.0_entropyScale/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
      --save_result_path ${logdir}/${save_name}/predict/model${t}_scale10.0_entropyScale/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
      --batch_size 256

done

