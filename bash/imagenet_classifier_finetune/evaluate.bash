EVALUATE_FLAGS="
    --ref_batch /mnt/data1/shengming/guided_ddpm/pretrain/VIRTUAL_imagenet256_labeled.npz
    --sample_batch /mnt/data1/shengming/guided_ddpm/log/imagenet_classifier_finetune/predict/test0/samples_300000x256x256x3.npz
    --save_result_path /mnt/data1/shengming/guided_ddpm/log/imagenet_classifier_finetune/reuslt_dict.yaml
    "

python evaluations/evaluator.py \
      $EVALUATE_FLAGS

