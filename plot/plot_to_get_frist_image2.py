import matplotlib.pyplot as plt
import yaml
import os

import zipfile
import numpy as np

# ['all_grad_norm', 'all_updated_grad_norm', 'all_probability', 'all_entropy', 'all_entropy_scale', 'all_probability_distribution']


images = {}
probability = {}
grad_norm = {}
updated_grad_norm = {}
for method in ['0-640_no_grad', 'baseline', 'ECT+EDS']:
    npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_first_image/predict/model500000_stepsddim25_scale10.0_sample50000/class14/metainfo_{}.npz'.format(method)
    npz = np.load(npz_path)
    images[method] = (npz['batch_image'] / 2 + 0.5).transpose(0, 2, 3, 1)  # (B,H,W,C)
    probability[method] = npz['batch_probability'].transpose(1, 0)
    grad_norm[method] = npz['batch_grad_norm'].transpose(1, 0)
    updated_grad_norm[method] = npz['batch_updated_grad_norm'].transpose(1, 0)

# os.makedirs('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_first_image/predict/model500000_stepsddim25_scale10.0_sample50000/class14/imgs',exist_ok=True)
# for image_id in range(images['baseline'].shape[0]):
#     print('image_id', image_id)
#     fig = plt.figure(figsize=(40, 10), linewidth=2)
#     plt.axis('off')
#
#     for idx, method in enumerate(['0-640_no_grad', 'baseline',  'ECT+EDS']):
#         ax = plt.subplot(1, 3, idx + 1)
#         ax.set_title(method)
#         ax.axis('off')
#         plt.imshow(
#             images[method][image_id]
#         )
#
#     save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_first_image/predict/model500000_stepsddim25_scale10.0_sample50000/class14/imgs/img_{}'.format(image_id)
#     plt.savefig(save_path, bbox_inches='tight')
#     # plt.show()

selected_id = 11
# os.makedirs('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_first_image/predict/model500000_stepsddim25_scale10.0_sample50000/class14/imgs',exist_ok=True)
# for image_id in [selected_id]:
#     print('image_id', image_id)
#
#     for idx, method in enumerate(['0-640_no_grad', 'baseline',  'ECT+EDS']):
#         fig = plt.figure(linewidth=2, dpi=600)
#         plt.axis('off')
#         plt.imshow(
#             images[method][image_id]
#         )
#
#         save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_first_image/predict/model500000_stepsddim25_scale10.0_sample50000/class14/imgs/img_{}_{}.pdf'.format(image_id, method)
#         plt.savefig(save_path, bbox_inches='tight')


fontdict = {
    'fontsize': 14,
    'family': 'Times New Roman',
    'weight': 'light',
}

import seaborn as sns

sns.set_theme(style="white")
sns.despine()
fontsize = 16

# grad norm
for method in ['baseline', 'ECT+EDS']:
    fig = plt.figure(figsize=(14, 3), linewidth=4)

    ax = fig.add_subplot(111)
    plt.xticks([960 - t for t in range(960, -1, -160)], [t for t in range(960, -1, -160)], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_ylabel('Gradient Norm', fontsize=fontsize + 2)
    ax.set_xlabel('Time Diffusion Steps', fontsize=fontsize + 2)
    if method == 'ECT+EDS':
        ax.set_ylim(0, 11.5)
        ax.plot([t for t in range(960, -1, -40)], updated_grad_norm[method][selected_id], lw=3, marker='*', color='red', label='gradient norm')
        # ax.annotate('point that the conditional guidance vanishes', xy=(960-640, 0.1), xytext=(960-600, 1), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=fontsize)
    else:
        ax.set_ylim(0, 11.5)
        ax.plot([t for t in range(960, -1, -40)], grad_norm[method][selected_id], lw=3, marker='*', color='red', label='gradient norm')
        ax.annotate('point that the conditional guidance vanishes', xy=(960-680, 0.1), xytext=(960-600, 1), arrowprops=dict(facecolor='black', shrink=0.05), fontsize=fontsize+4)
    plt.legend(loc='upper left', fontsize=fontsize+2)

    # probability
    ax2 = ax.twinx()
    ax2.plot([t for t in range(960, -1, -40)], probability[method][selected_id], lw=3, marker='o', color='blue', label='probability')
    ax2.set_ylabel('Probability', fontsize=fontsize + 2)
    ax2.set_ylim(0, 1.3)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize+2)

    plt.savefig(
        '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/grad_norm_and_probability_{}.pdf'.format(method),
        bbox_inches='tight'
    )

print('ok')
