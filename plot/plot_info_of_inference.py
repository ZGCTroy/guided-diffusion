import matplotlib.pyplot as plt
import yaml
import os

num_samples = 192
iteration = 500000
use_entropy_scale = False
steps = 'ddim25'
scale = 10.0
workspace = "/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion"
log_dir = os.path.join(workspace, 'log')
log_name = 'imagenet1000_classifier256x256_channel128_upperbound'
# predict_name = 'model{}_imagenet1000_steps{}_sample{}_plot'.format(iteration, steps, num_samples)
predict_name = 'conditional_model{}_imagenet1000_steps{}_getClassifierGroundthGuidance'.format(iteration, steps)
log_dir = os.path.join(workspace, 'log', log_name, 'predict',predict_name)
npz_path = os.path.join(log_dir, 'metainfo_scale{}_steps{}_samples_{}x256x256x3.npz'.format(scale, steps, num_samples))

npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/conditional_model500000_imagenet1000_stepsddim25_getClassifierGroundthGuidance/metainfo_scale10.0_stepsddim25_samples_192x3x256x256.npz'
# inference_mode = 'entropy_scale_inference'
# inference_mode = 'normal_inference'
inference_mode = 'ground_truth'

import zipfile
import numpy as np

npz = np.load(npz_path)
print(npz)
print(list(npz.keys()))
# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']

for name in list(npz.keys()):

    arr = npz[name]
    print(name, arr.shape)

    if name != 'all_probability_distribution':

        legends = []

        plt.figure()
        plt.ylabel(name[4:])
        plt.subplot(2, 2, 1)
        plt.plot(np.mean(arr, axis=1)[::-1])
        plt.plot(np.std(arr, axis=1)[::-1])
        plt.plot(np.median(arr, axis=1)[::-1])
        plt.legend(['mean', 'std', 'median'])

        plt.subplot(2, 2, 2)
        plt.plot(np.mean(arr, axis=1)[::-1])
        plt.legend(['mean'])

        plt.subplot(2, 2, 3)
        plt.plot(np.std(arr, axis=1)[::-1])
        plt.legend(['std'])

        plt.subplot(2, 2, 4)
        plt.plot(np.median(arr, axis=1)[::-1])
        plt.legend(['median'])

        plt.xlabel('Time diffusion steps from 1000 to 1')

        img_path = os.path.join(workspace, 'imgs/{}/{}/{}.png'.format(inference_mode, scale, name[4:]))
        plt.savefig(img_path)
        plt.show()
        plt.close()

    else:
        pass

