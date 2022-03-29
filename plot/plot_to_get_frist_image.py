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

images = {}
probability = {}
grad_norm = {}
for method in ['0-640_no_grad', 'baseline', 'ECT+EDS']:
    npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/get_first_image/predict/model500000_stepsddim25_scale10.0_sample50000/class14/metainfo_{}.npz'.format(method)
    npz = np.load(npz_path)
    images[method] = (npz['batch_image'] / 2 + 0.5).transpose(0, 2, 3, 1)  # (B,H,W,C)
    probability[method] = npz['batch_probability'].transpose(1,0)
    grad_norm[method] = npz['batch_grad_norm'].transpose(1,0)

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

fig = plt.figure(figsize=(20,15),linewidth=2, dpi=600)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0, hspace=0.5)

markers = ['*','o']
# grad norm
ax = plt.subplot(2, 2, 1)

ax.set_ylabel('Gradient Norm', fontdict=fontdict)
plt.xticks([t+1 for t, _ in enumerate(range(960, 480, -40))],[t for t in range(960, 480, -40)])

for idx, method in enumerate(['baseline', 'ECT+EDS']):
    ax.plot(grad_norm[method][selected_id][::-1][:12], lw=1,  label=method, marker=markers[idx])
    ax.legend()

ax = plt.subplot(2, 2, 2)
ax.yaxis.set_ticks_position('right')
ax.set_ylabel('Gradient Norm', fontdict=fontdict)
ax.yaxis.set_label_position('right')
plt.xticks([t+1 for t, _ in enumerate(range(480, -1, -40))],[t for t in range(480, -1, -40)])
for idx, method in enumerate(['baseline', 'ECT+EDS']):

    ax.plot(grad_norm[method][selected_id][::-1][12:], lw=1,  label=method, marker=markers[idx])
    ax.legend()

# probability
ax = plt.subplot(2, 2, 3)

ax.set_xlabel('Time Diffusion Steps', fontdict=fontdict)
ax.set_ylabel('Probability', fontdict=fontdict)
plt.xticks([t+1 for t, _ in enumerate(range(960, 480, -40))],[t for t in range(960, 480, -40)])

for idx, method in enumerate(['baseline', 'ECT+EDS']):
    ax.plot(probability[method][selected_id][::-1][:12], lw=1,  label=method, marker=markers[idx])
    ax.legend()

ax = plt.subplot(2, 2, 4)
ax.yaxis.set_ticks_position('right')
ax.set_xlabel('Time Diffusion Steps', fontdict=fontdict)
ax.set_ylabel('Probability', fontdict=fontdict)
ax.yaxis.set_label_position('right')
plt.xticks([t+1 for t, _ in enumerate(range(480, -1, -40))],[t for t in range(480, -1, -40)])

for idx, method in enumerate(['baseline', 'ECT+EDS']):
    ax.plot(probability[method][selected_id][::-1][12:], lw=1,  label=method, marker=markers[idx])
    ax.legend()


plt.tight_layout()
plt.savefig(
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/first_image2.png',
    bbox_inches='tight'
)


# plt.show()

print('ok')
