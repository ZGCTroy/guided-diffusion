# sample 10000 images from tinyimage dataset and save them into npy format(numpy format)

# from guided_diffusion.image_datasets import *
import argparse
import numpy as np
import torch as th
from tqdm import tqdm

from guided_diffusion.image_datasets import load_data
from torchvision import utils
import torch
import os
def main():
    npz_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/pretrain_model/VIRTUAL_imagenet-1000_256x256_labeled_sample50_class0-999.npz'
    npz = np.load(npz_path)
    img_arr = npz['arr_0']
    label_arr = npz['arr_1']
    print(img_arr.shape, label_arr.shape)
    os.makedirs('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/ref_batch',exist_ok=True)
    for class_id in range(1000):
        imgs_for_this_class = img_arr[label_arr == class_id]
        save_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/ref_batch/class{}.png'.format(class_id)
        print(imgs_for_this_class.shape)
        utils.save_image(
            torch.Tensor(imgs_for_this_class.transpose(0,3,1,2)[:9]),
            save_path,
            nrow=3,
            normalize=True,
            range=(0, 250),
        )


if __name__ == '__main__':
    main()
