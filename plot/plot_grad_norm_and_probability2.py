import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']

# fontdict = {
#     'fontsize': 16,
#     'family': 'Times New Roman',
#     'weight': 'light',
# }
import seaborn as sns
sns.set_theme(style="white")
sns.despine()
fontsize = 22
fig = plt.figure(figsize=(20, 14), linewidth=2, dpi=600)

npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample5000_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0-999_samples_5000x256x256x3.npz'
npz = np.load(npz_path)

# grad norm
arr = npz['all_grad_norm']
ax = fig.add_subplot(111)
ax.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1), lw=2, marker='*',
        color='red', label='gradient norm')
plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.set_ylabel('Gradient Norm', fontsize=fontsize + 2)
ax.set_xlabel('Time Diffusion Steps', fontsize=fontsize + 2)
ax.set_ylim(0,8)
ax.annotate('point that the conditional guidance vanishes', xy=(480, 0.1), xytext=(500, 1),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=fontsize)
plt.legend(loc='upper left',fontsize=fontsize)

# probability
arr = npz['all_probability']
ax2 = ax.twinx()
ax2.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1), lw=2, marker='o',
         color='blue', label='probability')
ax2.set_ylabel('Probability', fontsize=fontsize + 2)
ax2.set_ylim(0,1.1)
plt.yticks(fontsize=fontsize)

plt.legend(loc='upper right',fontsize=fontsize)

plt.savefig(
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/grad_norm_and_probability_ECT.pdf',
    bbox_inches='tight'
)
# plt.show()
plt.close()
print('ok')
