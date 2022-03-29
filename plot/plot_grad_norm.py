import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']


# plt.figure()
#
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot/metainfo_scale10.0_stepsddim25_class0_samples_192x256x256x3.npz'
# npz = np.load(npz_path)
# arr = npz['all_grad_norm']
# plt.subplot(3, 1, 1)
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.plot([t for t in range(960, -1, -40)], np.std(arr, axis=1))
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.xticks([960 - t for t in range(960, -1, -160)], [])
# plt.legend(['mean', 'std', 'median'])
# plt.ylabel('gradient norm')
# plt.xlabel('Time diffusion steps')
# plt.ylim(0, 20)
# plt.title('normal inference')
#
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_getClassifierGroundthGuidance/metainfo_scale10.0_stepsddim25_class0-0_samples_192x3x256x256.npz'
# npz = np.load(npz_path)
# arr = npz['all_grad_norm']
# plt.subplot(3, 1, 2)
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.plot([t for t in range(960, -1, -40)], np.std(arr, axis=1))
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.legend(['mean', 'std', 'median'])
# plt.ylabel('gradient norm')
# plt.ylim(0, 20)
# plt.title('ground truth inference')
# plt.xticks([960 - t for t in range(960, -1, -160)], [])
#
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0_samples_192x256x256x3.npz'
# npz = np.load(npz_path)
# arr = npz['all_updated_grad_norm']
# plt.subplot(3, 1, 3)
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.plot([t for t in range(960, -1, -40)], np.std(arr, axis=1))
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.legend(['mean', 'std', 'median'])
# plt.ylabel('gradient norm')
# plt.xlabel('Time diffusion steps')
# plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])
# plt.ylim(0, 20)
# plt.title('entropy scale inference')
#
# plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])
# plt.show()
#
# workspace = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion'
# img_path = os.path.join(workspace, 'imgs/grad_norm.png')





fig = plt.figure(figsize=(20,10))

# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot/metainfo_scale10.0_stepsddim25_class0_samples_192x256x256x3.npz'
# npz = np.load(npz_path)
# arr = npz['all_grad_norm']
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.xticks([960 - t for t in range(960, -1, -160)], [])
#
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_getClassifierGroundthGuidance/metainfo_scale10.0_stepsddim25_class0-0_samples_192x3x256x256.npz'
# # npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_getClassifierGroundthGuidance/metainfo_scale10.0_stepsddim25_class0-999_samples_5000x3x256x256.npz'
# npz = np.load(npz_path)
# arr = npz['all_grad_norm']
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.xticks([960 - t for t in range(960, -1, -160)], [])

# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot_entropyScale/metainfo_scale5.0_stepsddim25_class0_samples_192x256x256x3.npz'
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample192_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0_samples_192x256x256x3.npz'
npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample5000_plot_entropyScale/metainfo_scale10.0_stepsddim25_class0-999_samples_5000x256x256x3.npz'
npz = np.load(npz_path)
arr = npz['all_updated_grad_norm']
ax = plt.subplot(1,2,1)
ax.set_title('grad norm')
plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])

plt.ylabel('gradient norm')
plt.xlabel('Time diffusion steps')
plt.legend(['Baseline'])

arr = npz['all_updated_grad_norm']
ax = plt.subplot(1,2,1)
ax.set_title('grad norm')
plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)])

# plt.legend(['normal inference for class0', 'ground truth inference for class0', 'entropy scale inference for class0'])
plt.ylabel('gradient norm')
plt.xlabel('Time diffusion steps')
plt.legend(['Baseline'])


plt.savefig('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/grad_norm_and_probability.pdf')
plt.show()




# plt.figure()
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_getClassifierGroundthGuidance/metainfo_scale10.0_stepsddim25_class0-0_samples_192x3x256x256.npz'
# npz = np.load(npz_path)
# arr = npz['all_grad_norm']
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.xticks([960 - t for t in range(960, -1, -160)], [])
#
# npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_getClassifierGroundthGuidance/metainfo_scale10.0_stepsddim25_class0-999_samples_5000x3x256x256.npz'
# npz = np.load(npz_path)
# arr = npz['all_grad_norm']
# plt.plot([t for t in range(960, -1, -40)], np.mean(arr, axis=1))
# plt.xticks([960 - t for t in range(960, -1, -160)], [])
#
# plt.legend(['ground truth inference for class0-0', 'ground truth inference for class 0-999'])
# plt.ylabel('gradient norm')
# plt.xlabel('Time diffusion steps')
# plt.show()
#
# print(np.log(1000))

