str = print('123')

print(str)
# import numpy as np
# import torch
# a = torch.tensor([[ 1,  1]])
# b = torch.tensor([[3,3]])
# # print( torch.cdist(a.float(), b.float()))
#
# print(np.log(200))
# data = np.load('/mnt/data1/shengming/guided_ddpm/pretrain/admnet_imagenet256.npz')
# data = np.load('/mnt/data1/shengming/guided_ddpm/pretrain/VIRTUAL_imagenet256_labeled.npz')
# print(data['arr_0'].shape, data['arr_1'].shape)

# import blobfile as bf
# import os

# def _list_image_files_recursively(data_dir):
#     results = []
#     for entry in sorted(bf.listdir(data_dir)):
#         full_path = bf.join(data_dir, entry)
#         ext = entry.split(".")[-1]
#         if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
#             results.append(full_path)
#         elif bf.isdir(full_path):
#             results.extend(_list_image_files_recursively(full_path))
#     return results
#
# data_dir = '/mnt/data2/shengming/ImageNet/train'
#
# all_files = _list_image_files_recursively(data_dir)
#
# class_names = list(set([os.path.dirname(path).split("/")[-1] for path in all_files]))
#
# print(len(class_names), class_names[:10])


# from guided_diffusion import dist_util
# from mpi4py import MPI
# visible_gpus_list = []
# dist_util.setup_dist(visible_gpu_list=visible_gpus_list)
#
#
# from evaluations.evaluator import quick_compute_stats, quick_evaluate
#
# sample_path = '/mnt/data1/shengming/guided_ddpm/log/imagenet_classifier_baseline/predicted_imgs/images40000.npz'
# path = '/mnt/data1/shengming/guided_ddpm/pretrain/VIRTUAL_tinyimagenet256_stats10000.npz'
# # quick_compute_stats((path))
# # ref_stats, ref_stats_spatial = np.load('VIRTUALtinyimagenet-stats10000.npz')
#
# # print(ref_stats, ref_stats_spatial)
# quick_evaluate(sample_batch=sample_path, ref_stats_path=path)
