import torch as th
# import torch.distributed as dist
import torch.nn.functional as F
# from . import dist_util, logger
import numpy as np
import os
from torchvision import utils
# FID 14.
base_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample50000/scale10.0_stepsddim25_samples_50000x256x256x3.npz'
# FID 8.
entropy_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2/predict/model500000_stepsddim25_scale6.0_sample50000_entropyScale/scale6.0_stepsddim25_samples_50000x256x256x3.npz'

def get_data(path):
    data = np.load(path)
    img, label = data['arr_0'], data['arr_1']
    # print(img.shape, label.shape,'label img')
    return img,label

# baseline_data = np.load(base_path)
baseline_img, baseline_label = get_data(base_path)
# baseline_data['arr_0'], baseline_data['arr_1']

# entropy_data = np.load(entropy_path)
entropy_img, entropy_label = get_data(entropy_path)
# entropy_data['arr_0'], entropy_data['arr_1']

selected_class = [12,3,407,294,323,873,970]

baseline = [1,2,4,4,2,2,3]
entropy = [3,4,3,3,3,3,3]
column_num = 5
upper_bound = 21

def save_samples(baseline_label=baseline_label,
                baseline_img=baseline_img,
                entropy_label=entropy_label,
                entropy_img=entropy_img,

                per_class_ub=upper_bound,
                save_batch_size=32,    
                nrow=column_num,):

    baseline_saved, entropy_saved = [], []
    
    for line_idx,i in enumerate(selected_class):
        idx = np.where(baseline_label==i)[0]
        if len(idx) < per_class_ub:
            continue
        idx = idx[:per_class_ub]
        baseline_batch = baseline_img[idx].transpose(0, 3, 1, 2)
        baseline_batch = 2 * (th.from_numpy(baseline_batch) / 255.) - 1.

        idx = np.where(entropy_label==i)[0]
        if len(idx) < per_class_ub:
            continue
        idx = idx[:per_class_ub]
        entropy_batch = entropy_img[idx].transpose(0, 3, 1, 2)
        entropy_batch = 2 * (th.from_numpy(entropy_batch) / 255.) - 1.

        baseline_saved.append(baseline_batch[(baseline[line_idx]-1)*column_num:baseline[line_idx]*column_num+1])
        entropy_saved.append(entropy_batch[(entropy[line_idx]-1)*column_num:entropy[line_idx]*column_num+1])

    baseline_saved = th.cat(baseline_saved, dim=0)
    entropy_saved = th.cat(entropy_saved, dim=0)

    print(baseline_saved.shape, entropy_saved.shape)

    utils.save_image(
        baseline_saved,
        f"basleine.pdf",
        nrow=nrow+1,
        normalize=True,
        range=(-1, 1),
    )
    utils.save_image(
        entropy_saved,
        f"entropy.pdf",
        nrow=nrow+1,
        normalize=True,
        range=(-1, 1),
    )
    return 

save_samples()

# for x in selected_class:
#     idx = np.where(baseline_label == x)
#     baseline_batch = baseline_img[idx][:upper_bound]
    
#     idx = np.where(entropy_label == x)
#     entropy_batch = entropy_img[idx][:upper_bound]

#     print(baseline_batch.shape, entropy_batch.shape)
# print(img.shape, label.shape)