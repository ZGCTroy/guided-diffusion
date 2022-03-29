import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np
import torch
from torchvision import utils

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']

fontdict = {
    'fontsize': 20,
    'family': 'Times New Roman',
    'weight': 'light',
}

images = {}
probability = {}
grad_norm = {}

selected_image_ids = {
    628: [19, 6, 13, 8],
    417: [0, 4, 9, 16],
    449: [2,6,23,12],
    508: [3,7,0,17]
}

os.makedirs('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img_steps250', exist_ok=True)
save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img_steps250/compared_img.pdf'

total_images = {
    'baseline': [],
    'ECT+EDS': []
}
for key, value in selected_image_ids.items():

    for method in ['baseline', 'ECT+EDS']:
        npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_compared_image/predict/model500000_steps250_scale10.0_sample50000/class{}/metainfo_{}.npz'.format(key,
                                                                                                                                                                                             method)
        npz = np.load(npz_path)
        images[method] = npz['batch_image']  # (B,H,W,C)
        print(images[method].shape)
        for id in value:
            total_images[method].append(torch.Tensor(images[method][id]))



i = 0
j = 0

final_images = []
save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img_steps250/{}.pdf'.format( 'baseline')
for key, value in selected_image_ids.items():

    for _ in range(4):
        print(i)
        final_images.append(total_images['baseline'][i])
        i += 1

utils.save_image(
    final_images,
    save_path,
    nrow=4,
    normalize=True,
    range=(-1, 1),
)

final_images = []
save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img_steps250/{}.pdf'.format( 'ECT+EDS')
for key, value in selected_image_ids.items():

    for _ in range(4):
        final_images.append(total_images['ECT+EDS'][j])
        j += 1

utils.save_image(
    final_images,
    save_path,
    nrow=4,
    normalize=True,
    range=(-1, 1),
)
