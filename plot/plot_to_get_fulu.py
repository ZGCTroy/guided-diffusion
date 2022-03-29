import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np
import torch
from torchvision import utils
import torch




class_ids = [1, 279, 323,386,130,852,933,562,417,281,90,992] # 1金鱼，2

os.makedirs('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/fulu', exist_ok=True)

npz_paths = [
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2/predict/conditional_model500000_steps250_scale1.0_entropyScale/scale1.0_steps250_samples_50000x256x256x3.npz',
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2/predict/conditional_model500000_stepsddim25_scale2.0_sample50000_entropyScale/scale2.0_stepsddim25_samples_50000x256x256x3.npz',
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier128x128_channel128_baseline/predict/conditional_model300000_steps250_scale0.4_sample50000_entropyScale/scale0.4_steps250_samples_50000x128x128x3.npz',
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier64x64_channel128_baseline/predict/conditional_model300000_steps250_scale0.1_sample50000_entropyScale/scale0.1_steps250_samples_50000x64x64x3.npz'

]
save_paths = [
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/fulu/256x256_CADM-G+ECT+EDS_steps250.pdf',
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/fulu/256x256_CADM-G+ECT+EDS_stepsddim25.pdf',
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/fulu/128x128_CADM-G+ECT+EDS_steps250.pdf',
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/fulu/64x64_CADM-G+ECT+EDS_steps250.pdf'
]

for npz_path, save_path in zip(npz_paths, save_paths):
    npz = np.load(npz_path)
    images = npz['arr_0']
    labels = npz['arr_1']
    total_images = []
    for class_id in class_ids:
        total_images.append(torch.Tensor(images[labels == class_id])[:8])

    total_images = torch.cat(total_images, dim=0)
    total_images = total_images.permute(0,3,1,2)
    total_images = 2 *(total_images / 255.0) - 1
    print(total_images.shape)
    utils.save_image(
        total_images,
        save_path,
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )
