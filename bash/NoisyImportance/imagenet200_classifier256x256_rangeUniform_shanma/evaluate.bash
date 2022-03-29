EVALUATE_FLAGS="
    --ref_batch /workspace/guided-diffusion/pretrain_model/VIRTUAL_tinyimagenet256_labeled.npz
    --sample_batch /workspace/guided-diffusion/log/imagenet200_classifier256x256_channel64_rangeUniform/predict/model010000/samples_1000x256x256x3.npz
    --save_result_path /workspace/guided-diffusion/log/imagenet200_classifier256x256_channel64_rangeUniform/predict/model010000/result.yaml
    --batch_size 32
    "

CUDA_VISIBLE=0,1 python evaluations/evaluator.py \
      $EVALUATE_FLAGS

