
save_name="imagenet200_classifier256x256_channel64_baseline"
num_samples="1000"
scale="10.0"
steps="ddim25"

for ((i=290000; i>=290000; i=i-30000))
do
    t=$(printf "%06d" $i)

    echo $t

    CUDA_VISIBLE=0,1 python evaluations/evaluator.py \
      --ref_batch /workspace/guided-diffusion/pretrain_model/VIRTUAL_tinyimagenet256_labeled10000.npz \
      --sample_batch /workspace/guided-diffusion/log/${save_name}/predict/model${t}_condrange750_scale0.0/scale${scale}_steps${steps}_samples_${num_samples}x256x256x3.npz \
      --save_result_path /workspace/guided-diffusion/log/${save_name}/predict/model${t}_condrange750_scale0.0/result_scale${scale}_steps${steps}_sample${num_samples}.yaml \
      --batch_size 16

done
