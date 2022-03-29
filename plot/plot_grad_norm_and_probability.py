import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']

fontdict = {
    'fontsize': 20,
    'family': 'Times New Roman',
    'weight': 'light',
}

fig = plt.figure(figsize=(10, 20), linewidth=2, dpi=600)

npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample5000_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0-999_samples_5000x256x256x3.npz'
npz = np.load(npz_path)

# grad norm
arr = npz['all_grad_norm']
ax = plt.subplot(2, 1, 1)
ax.set_title('gradient norm')
plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1), lw=2)
plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])

plt.ylabel('gradient norm', fontdict=fontdict)
plt.xlabel('Time diffusion steps', fontdict=fontdict)
plt.legend(['Baseline'])

# probability
arr = npz['all_probability']
ax = plt.subplot(2, 1, 2)
ax.set_title('probability')
plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1), lw=2)
plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])
plt.ylabel('probability', fontdict=fontdict)
plt.xlabel('Time diffusion steps', fontdict=fontdict)
plt.legend(['Baseline'])

plt.savefig('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/grad_norm_and_probability.pdf')
plt.show()
