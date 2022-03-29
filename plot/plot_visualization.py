import matplotlib.pyplot as plt
import yaml
import os
from guided_diffusion.sample_util import save_samples
import zipfile
import numpy as np

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']

fontdict = {
    'fontsize': 20,
    'family': 'Times New Roman',
    'weight': 'light',
}

fig = plt.figure(figsize=(10, 20), linewidth=2, dpi=600)

workspace = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion'
steps = 250

# baseline
# log_dir = os.path.join(workspace, 'pretrain_model')
# baseline_npz_path = os.path.join(log_dir, 'admnet_guided_imagenet256.npz')
# npz = np.load(baseline_npz_path)
# arr = npz['arr_0']
# label_arr = npz['arr_1']
# ax = plt.subplot(2, 1, 1)
# ax.set_title('baseline')
#
# sample_dir = "ADM-G_baseline/images256x256_" + "unconditional_scale{}_steps{}_sample{}".format(10.0, 250, 50000)
# save_samples(arr, label_arr, 1000, log_dir=log_dir, sample_dir=sample_dir)

# entropyScale
log_dir = os.path.join(workspace, 'log/imagenet1000_classifier256x256_channel128_baseline/predict/model500000_steps250_scale6.0_sample50000_entropyScale')
baseline_npz_path = os.path.join(log_dir, 'scale6.0_steps250_samples_50000x256x256x3.npz')
npz = np.load(baseline_npz_path)
arr = npz['arr_0']
label_arr = npz['arr_1']
ax = plt.subplot(2, 1, 1)
ax.set_title('baseline')

sample_dir = "images_" + "scale{}_steps{}_sample{}".format(10.0, 250, 50000)
save_samples(arr, label_arr, 1000, log_dir=log_dir, sample_dir=sample_dir)
