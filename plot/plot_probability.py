import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']



plt.figure()

npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot/metainfo_scale10.0_stepsddim25_class0_samples_192x256x256x3.npz'
npz = np.load(npz_path)
arr = npz['all_probability']
plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
plt.xticks([960 - t for t in range(960, -1, -160)], [])


npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot_entropyScale/metainfo_scale5.0_stepsddim25_class0_samples_192x256x256x3.npz'
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0_samples_192x256x256x3.npz'
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample5000_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0-999_samples_5000x256x256x3.npz'
npz = np.load(npz_path)
arr = npz['all_probability']
plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])

plt.legend(['normal inference for class0', 'entropy scale inference for class0'])
plt.ylabel('probability')
plt.xlabel('Time diffusion steps')

plt.savefig('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/probability.pdf')
plt.show()




