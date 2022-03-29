workspace_dir=/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion
logdir=${workspace_dir}/log
pretrain_model=${workspace_dir}/pretrain_model

save_name="imagenet1000_conditionalDDPM_classifier256x256_channel128_upperbound"
num_samples="50000"
scale="2.5"
steps="ddim25"

for ((i=500000; i>=500000; i=i-30000))
do
    t=$(printf "%06d" $i)

    echo $t

  CUDA_VISIBLE=0,1,2,3,4,5,6,7,8 python evaluations/evaluator.py \
        --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
        --sample_batch ${logdir}/${save_name}/predict/model${t}_imagenet1000_stepsddim25_sample50000/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
        --save_result_path ${logdir}/${save_name}/predict/model${t}_imagenet1000_stepsddim25_sample50000/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
        --batch_size 256
#
#  CUDA_VISIBLE=0,1,2,3,4,5,6,7,8 python evaluations/evaluator.py \
#        --ref_batch ${pretrain_model}/VIRTUAL_imagenet256_labeled.npz \
#        --sample_batch ${logdir}/${save_name}/predict/model${t}_imagenet1000_stepsddim25_sample50000_entropyScale/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
#        --save_result_path ${logdir}/${save_name}/predict/model${t}_imagenet1000_stepsddim25_sample50000_entropyScale/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
#        --batch_size 256

done