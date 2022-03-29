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
    1: [4, 10, 23],
    13: [22, 17, 19],
    22: [6, 3, 13],
    5: [5, 7, 17],
    984: [23, 16, 0],
    985: [18, 12, 8],
    970: [5, 6, 16],
    874:[6,21, 0], # 公交车，巴士
    923: [0,1,12],
    488:[5,22,6], # 锁链
}

os.makedirs('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img',exist_ok=True)
save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img/compared_img.pdf'

total_images = {
    'baseline': [],
    'ECT+EDS': []
}
for key, value in selected_image_ids.items():

    for method in ['baseline', 'ECT+EDS']:
        npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_compared_image/predict/model500000_stepsddim25_scale10.0_sample50000/class{}/metainfo_{}.npz'.format(key,
                                                                                                                                                                                                method)
        npz = np.load(npz_path)
        images[method] = npz['batch_image']  # (B,H,W,C)
        print(images[method].shape)
        for id in value:
            total_images[method].append(torch.Tensor(images[method][id]))

final_images = []
num = len(total_images['baseline'])
i = 0
j = 0
while i < num:
    print(i, num)
    for _ in range(3 * 2):
        final_images.append(total_images['baseline'][i])
        i += 1

    for _ in range(3 * 2):
        final_images.append(total_images['ECT+EDS'][j])
        j += 1

utils.save_image(
    final_images,
    save_path,
    nrow=6,
    normalize=True,
    range=(-1, 1),
)

print('ok1')


i = 0
j = 0

for key, value in selected_image_ids.items():
    save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/compared_img/class{}.pdf'.format(key)

    final_images = []

    for _ in range(3):
        final_images.append(total_images['baseline'][i])
        i += 1

    for _ in range(3):
        final_images.append(total_images['ECT+EDS'][j])
        j += 1

    utils.save_image(
        final_images,
        save_path,
        nrow=3,
        normalize=True,
        range=(-1, 1),
    )