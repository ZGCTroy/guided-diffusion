# sample 10000 images from tinyimage dataset and save them into npy format(numpy format)

# from guided_diffusion.image_datasets import *
import argparse
import numpy as np
import torch as th
from tqdm import tqdm

from guided_diffusion.image_datasets import load_data


def main():
    workspace = "/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion"
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/workspace/mnt/storage/yangdecheng/imagenet_1k/ImageNet-1k/train")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset_type', type=str, default="imagenet-1000")
    args = parser.parse_args()

    img_size = 256
    data_type = 'imagenet-1000'
    num_sample = 50

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=img_size,
        class_cond=True,
        random_crop=True,
        dataset_type=data_type,
        used_attributes='',
        tot_class=1000,
    )


    sample_lst = [[] for i in range(1000)]
    while True:
        sample_len = [ len(i) for i in sample_lst]
        is_end = True
        print(sample_len)
        for i in sample_len:
            if i < num_sample:
                is_end = False
        print(is_end)
        if is_end:
            break

        sample, extra = next(data)
        label = extra['y']

        sample = ((sample + 1.) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        label = label.numpy().reshape(-1)
        for idx, x in enumerate(label):
            if x in range(10):
                sample_lst[x].append(sample[idx].reshape(1,img_size,img_size,3))


    for idx, lst in enumerate(sample_lst):
        class_id = idx
        samples = np.concatenate(lst, axis=0)
        samples = samples[:num_sample]
        print(samples.shape)
        np.savez(f'{workspace}/pretrain_model/VIRTUAL_{data_type}_{img_size}x{img_size}_labeled_sample{num_sample}_class{class_id}.npz', samples)
        print('sample complete!', class_id, samples.shape)

    all_samples = []
    all_labels = []
    for idx, lst in enumerate(sample_lst):
        class_id = idx
        all_samples.extend(lst[:num_sample])
        all_labels.extend([class_id for i in range(num_sample)])

    arr = np.concatenate(all_samples, axis=0)
    label_arr = np.concatenate(all_labels, axis=0)
    np.savez(f'{workspace}/pretrain_model/VIRTUAL_{data_type}_{img_size}x{img_size}_labeled_sample{num_sample}_class0-999.npz', arr, label_arr)




if __name__ == '__main__':
    main()
