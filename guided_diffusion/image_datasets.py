import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torch.distributed as dist

def load_data(
        *,

        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=True,
        dataset_type="imagenet-1000",
        used_attributes="",
        tot_class=1000,
        imagenet200_class_list_file_path="",
        celeba_attribures_path=""
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param tot_class:
    :param used_attributes:
    :param dataset_type:
    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)
    classes = None

    if dataset_type == 'celebahq':
        with open(celeba_attribures_path, "r") as Attr_file:
            Attr_info = Attr_file.readlines()
            Attr_info = Attr_info[1:]

    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore. 
        # dir_path = '/mnt/data1/shengming/tiny-imagenet-200/train'
        # dir_list = os.listdir(dir_path)

        if dataset_type in ['imagenet-1000', 'imagenet-200']:
            def get_class(path):
                return os.path.dirname(path).split("/")[-1]

            if dataset_type == 'imagenet-200':
                with open(imagenet200_class_list_file_path, "r") as f:
                    used_class_list = f.readlines()
                    used_class_list = [p.strip() for p in used_class_list]
                all_files = [path for path in all_files if get_class(path) in used_class_list]

            class_names = [get_class(path) for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}

            if tot_class > 0:
                class_names = [x for x in class_names if sorted_classes[x] < tot_class]
                classes = [sorted_classes[x] for x in class_names]
                all_files = [path for path in all_files if sorted_classes[get_class(path)] < tot_class]
                assert len(all_files) == len(classes)

        elif dataset_type == 'celebahq':
            def get_attribute(path):
                """
                Given a image path, grab its corresponding attribute label
                """
                import os
                idx = int(os.path.basename(path)[:-4])
                attributes = Attr_info[idx].split()
                atts = []
                for num in used_attributes.split(','):
                    idx = int(num)
                    att = (float(attributes[idx]) + 1.) / 2.
                    atts.append(att)
                return atts

            classes = [get_attribute(path) for path in all_files]

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        dataset_type=dataset_type
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            dataset_type='imagenet-1000',

    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            if self.dataset_type in ['imagenet-1000', 'imagenet-200']:
                out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            elif self.dataset_type == 'celebahq':
                out_dict["y"] = np.array(self.local_classes[idx], dtype=np.float32)

        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
